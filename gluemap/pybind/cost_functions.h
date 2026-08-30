#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <stdexcept>

#include "vendor/colmap/estimators/cost_functions/reprojection_error.h"
#include "vendor/colmap/estimators/cost_functions/utils.h"

namespace colmap {

// The upstream COLMAP functor is intentionally omitted from the reduced
// vendored header. Fixed-lag BA needs the same fixed-pose topology as
// DefaultBundleAdjuster, so retain the exact two-parameter formulation here.
template <typename CameraModel>
class ReprojErrorConstantPoseCostFunctor
    : public AutoDiffCostFunctor<
          ReprojErrorConstantPoseCostFunctor<CameraModel>, 2, 3,
          CameraModel::num_params> {
public:
  ReprojErrorConstantPoseCostFunctor(const Eigen::Vector2d &point2D,
                                     const Rigid3d &cam_from_world)
      : cam_from_world_(cam_from_world), reproj_cost_(point2D) {}

  template <typename T>
  bool operator()(const T *const point3D_in_world,
                  const T *const camera_params, T *residuals) const {
    const Eigen::Matrix<T, 7, 1> cam_from_world =
        cam_from_world_.params.cast<T>();
    return reproj_cost_(point3D_in_world, cam_from_world.data(),
                        camera_params, residuals);
  }

private:
  const Rigid3d cam_from_world_;
  const ReprojErrorCostFunctor<CameraModel> reproj_cost_;
};

} // namespace colmap

// ----------------------------------------
// FejPosePriorCostFunction
// ----------------------------------------
// Dense pose-only square-root prior evaluated entirely in native code.  The
// parameter blocks use COLMAP's [x, y, z, w, tx, ty, tz] ambient layout and
// Ceres EigenQuaternionManifold tangent convention.
class FejPosePriorCostFunction final : public ceres::CostFunction {
public:
  FejPosePriorCostFunction(const Eigen::MatrixXd &factor,
                           const Eigen::VectorXd &factor_residual,
                           const Eigen::MatrixXd &linearization)
      : factor_(factor), factor_residual_(factor_residual),
        linearization_(linearization) {
    const Eigen::Index pose_count = linearization_.rows();
    if (pose_count < 1 || linearization_.cols() != 7 ||
        factor_.cols() != pose_count * 6 || factor_.rows() < 1 ||
        factor_residual_.size() != factor_.rows()) {
      throw std::invalid_argument("FEJ pose prior dimensions differ");
    }
    set_num_residuals(static_cast<int>(factor_.rows()));
    mutable_parameter_block_sizes()->assign(
        static_cast<size_t>(pose_count), 7);
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const Eigen::Index pose_count = linearization_.rows();
    Eigen::VectorXd delta(pose_count * 6);
    for (Eigen::Index index = 0; index < pose_count; ++index) {
      const double *current_value = parameters[index];
      const Eigen::Quaterniond current(current_value[3], current_value[0],
                                       current_value[1], current_value[2]);
      const Eigen::Quaterniond origin(
          linearization_(index, 3), linearization_(index, 0),
          linearization_(index, 1), linearization_(index, 2));
      const Eigen::Quaterniond difference = current * origin.conjugate();
      const Eigen::Vector3d imaginary = difference.vec();
      const double norm = imaginary.norm();
      if (norm == 0.0) {
        delta.segment<3>(index * 6).setZero();
      } else {
        delta.segment<3>(index * 6) =
            imaginary * (std::atan2(norm, difference.w()) / norm);
      }
      delta.segment<3>(index * 6 + 3) =
          Eigen::Map<const Eigen::Vector3d>(current_value + 4) -
          linearization_.row(index).segment<3>(4).transpose();
    }
    Eigen::Map<Eigen::VectorXd>(residuals, factor_.rows()) =
        factor_ * delta + factor_residual_;

    if (jacobians == nullptr) {
      return true;
    }
    for (Eigen::Index index = 0; index < pose_count; ++index) {
      if (jacobians[index] == nullptr) {
        continue;
      }
      const double *current = parameters[index];
      Eigen::Matrix<double, 4, 3> plus;
      plus << current[3], current[2], -current[1], -current[2],
          current[3], current[0], current[1], -current[0], current[3],
          -current[0], -current[1], -current[2];
      Eigen::Map<
          Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>>
          jacobian(jacobians[index], factor_.rows(), 7);
      jacobian.setZero();
      const auto tangent = factor_.middleCols(index * 6, 6);
      jacobian.leftCols<4>() = tangent.leftCols<3>() * plus.transpose();
      jacobian.rightCols<3>() = tangent.rightCols<3>();
    }
    return true;
  }

private:
  const Eigen::MatrixXd factor_;
  const Eigen::VectorXd factor_residual_;
  const Eigen::MatrixXd linearization_;
};

inline ceres::CostFunction *
CreateFejPosePriorCost(const Eigen::MatrixXd &factor,
                       const Eigen::VectorXd &factor_residual,
                       const Eigen::MatrixXd &linearization) {
  return new FejPosePriorCostFunction(factor, factor_residual, linearization);
}

// ----------------------------------------
// RotationGeodesicError
// ----------------------------------------
// Computes the geodesic error between rotation quaternions.
struct RotationGeodesicError
    : public colmap::AutoDiffCostFunctor<RotationGeodesicError, 3, 4, 4> {
public:
  explicit RotationGeodesicError(const Eigen::Vector4d &j_q_i)
      : j_q_i_(j_q_i) {}

  template <typename T>
  bool operator()(const T *const i_q_w, const T *const j_q_w,
                  T *residuals_ptr) const {
    const T w_q_j[4] = {j_q_w[0], -j_q_w[1], -j_q_w[2], -j_q_w[3]};

    T tmp_i_q_j[4];
    ceres::QuaternionProduct(i_q_w, w_q_j, tmp_i_q_j);

    T q_res[4];
    const Eigen::Matrix<T, 4, 1> j_q_i = j_q_i_.cast<T>();
    ceres::QuaternionProduct(j_q_i.data(), tmp_i_q_j, q_res);

    ceres::QuaternionToAngleAxis(q_res, residuals_ptr);

    return true;
  }

private:
  const Eigen::Vector4d j_q_i_;
};

// ----------------------------------------
// PairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i) is minimized.
struct PairwiseDirectionError
    : public colmap::AutoDiffCostFunctor<PairwiseDirectionError, 3, 3, 3, 1> {
  PairwiseDirectionError(const Eigen::Vector3d &translation_obs)
      : translation_obs_(translation_obs) {}

  template <typename T>
  bool operator()(const T *position1, const T *position2, const T *scale,
                  T *residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        translation_obs_.cast<T>() -
        scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                    Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
    return true;
  }

private:
  const Eigen::Vector3d translation_obs_;
};

// ----------------------------------------
// ReprojErrorCostWithNegativeDepthFunctor
// ----------------------------------------
// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
// This version handles negative depth (points behind camera).
template <typename CameraModel>
class ReprojErrorCostWithNegativeDepthFunctor
    : public colmap::AutoDiffCostFunctor<
          ReprojErrorCostWithNegativeDepthFunctor<CameraModel>, 2, 3, 7,
          CameraModel::num_params> {
public:
  explicit ReprojErrorCostWithNegativeDepthFunctor(
      const Eigen::Vector2d &point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T *const point3D, const T *const cam_from_world,
                  const T *const camera_params, T *residuals) const {
    Eigen::Matrix<T, 3, 1> point3D_in_cam =
        colmap::EigenQuaternionMap<T>(cam_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(cam_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

    // Always negate the point for negative depth projection
    point3D_in_cam = -point3D_in_cam;
    if (CameraModel::ImgFromCam(camera_params, point3D_in_cam[0],
                                point3D_in_cam[1], point3D_in_cam[2],
                                &residuals[0], &residuals[1])) {
      residuals_vec -= point2D_.cast<T>();
    } else {
      residuals_vec.setZero();
    }
    return true;
  }

private:
  const Eigen::Vector2d point2D_;
};
