#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/rotation.h>

#include "vendor/colmap/estimators/cost_functions/reprojection_error.h"
#include "vendor/colmap/estimators/cost_functions/utils.h"

// ----------------------------------------
// RotationGeodesicError
// ----------------------------------------
// Computes the geodesic error between rotation quaternions.
struct RotationGeodesicError
    : public colmap::AutoDiffCostFunctor<RotationGeodesicError, 3, 4, 4>
{
public:
  explicit RotationGeodesicError(const Eigen::Vector4d &j_q_i)
      : j_q_i_(j_q_i) {}

  template <typename T>
  bool operator()(const T *const i_q_w, const T *const j_q_w, T *residuals_ptr) const
  {
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
// BATAPairwiseDirectionError
// ----------------------------------------
// Computes the error between a translation direction and the direction formed
// from two positions such that t_ij - scale * (c_j - c_i) is minimized.
struct BATAPairwiseDirectionError
    : public colmap::AutoDiffCostFunctor<BATAPairwiseDirectionError, 3, 3, 3, 1>
{
  BATAPairwiseDirectionError(const Eigen::Vector3d &translation_obs)
      : translation_obs_(translation_obs) {}

  template <typename T>
  bool operator()(const T *position1,
                  const T *position2,
                  const T *scale,
                  T *residuals) const
  {
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
// ScaledObsPairwiseDirectionError
// ----------------------------------------
// Like BATAPairwiseDirectionError but the scale multiplies the observed
// translation instead of the position difference:
//   scale * t_ij - (c_j - c_i)
struct ScaledObsPairwiseDirectionError
    : public colmap::AutoDiffCostFunctor<ScaledObsPairwiseDirectionError, 3, 3, 3, 1>
{
  ScaledObsPairwiseDirectionError(const Eigen::Vector3d &translation_obs)
      : translation_obs_(translation_obs) {}

  template <typename T>
  bool operator()(const T *position1,
                  const T *position2,
                  const T *scale,
                  T *residuals) const
  {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        scale[0] * translation_obs_.cast<T>() -
        (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
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
    : public colmap::AutoDiffCostFunctor<ReprojErrorCostWithNegativeDepthFunctor<CameraModel>,
                                         2,
                                         3,
                                         7,
                                         CameraModel::num_params>
{
public:
  explicit ReprojErrorCostWithNegativeDepthFunctor(const Eigen::Vector2d &point2D)
      : point2D_(point2D) {}

  template <typename T>
  bool operator()(const T *const point3D,
                  const T *const cam_from_world,
                  const T *const camera_params,
                  T *residuals) const
  {
    Eigen::Matrix<T, 3, 1> point3D_in_cam =
        colmap::EigenQuaternionMap<T>(cam_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(cam_from_world + 4);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals_vec(residuals);

    // Always negate the point for negative depth projection
    point3D_in_cam = -point3D_in_cam;
    if (CameraModel::ImgFromCam(camera_params,
                                point3D_in_cam[0],
                                point3D_in_cam[1],
                                point3D_in_cam[2],
                                &residuals[0],
                                &residuals[1]))
    {
      residuals_vec -= point2D_.cast<T>();
    }
    else
    {
      residuals_vec.setZero();
    }
    return true;
  }

private:
  const Eigen::Vector2d point2D_;
};

// ----------------------------------------
// DepthRegularizationFunctor
// ----------------------------------------
// Cost function that penalizes deviations of depth (z in camera coordinates)
// from an initial depth value.
struct DepthRegularizationFunctor
    : public colmap::AutoDiffCostFunctor<DepthRegularizationFunctor, 1, 3, 7>
{
public:
  explicit DepthRegularizationFunctor(double initial_depth)
      : initial_depth_(initial_depth), initial_depth_inv_(1 / initial_depth) {}

  template <typename T>
  bool operator()(const T *const point3D,
                  const T *const cam_from_world,
                  T *residuals) const
  {
    Eigen::Matrix<T, 3, 1> point3D_in_cam =
        colmap::EigenQuaternionMap<T>(cam_from_world) *
            colmap::EigenVector3Map<T>(point3D) +
        colmap::EigenVector3Map<T>(cam_from_world + 4);

    T current_depth = point3D_in_cam[2];

    if (current_depth * initial_depth_ < T(1e-6)) {
      // If depth is very close to zero, or it is of opposite sign, set residual to zero to avoid instability
      residuals[0] = T(0);
      return true;
    }

    if (ceres::abs(current_depth) < T(1.0)) {
      residuals[0] = (current_depth - T(initial_depth_));
    } else {
      residuals[0] = (T(1.) / current_depth - T(initial_depth_inv_));
    }

    return true;
  }

private:
  const double initial_depth_;
  const double initial_depth_inv_;
};

// ----------------------------------------
// Point3DConsistencyError
// ----------------------------------------
// Cost function that enforces consistency between two 3D points after
// transformation by their respective scales and centers:
// point3d_1 / scale_1 + center_1 = point3d_2 / scale_2 + center_2
struct Point3DConsistencyError
    : public colmap::AutoDiffCostFunctor<Point3DConsistencyError, 3, 3, 3, 1, 1>
{
  Point3DConsistencyError(const Eigen::Vector3d &point3d_1,
                          const Eigen::Vector3d &point3d_2)
      : point3d_1_(point3d_1), point3d_2_(point3d_2) {}

  template <typename T>
  bool operator()(const T *center_1,
                  const T *center_2,
                  const T *scale_1,
                  const T *scale_2,
                  T *residuals) const
  {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        (point3d_1_.cast<T>() / scale_1[0] +
         Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center_1)) -
        (point3d_2_.cast<T>() / scale_2[0] +
         Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center_2));
    return true;
  }

private:
  const Eigen::Vector3d point3d_1_;
  const Eigen::Vector3d point3d_2_;
};

// ----------------------------------------
// Point3DConsistencySameCamError
// ----------------------------------------
// Cost function that enforces consistency between two 3D points with the same
// center but different scales:
// point3d_1 / scale_1 + center = point3d_2 / scale_2 + center
struct Point3DConsistencySameCamError
    : public colmap::AutoDiffCostFunctor<Point3DConsistencySameCamError, 3, 3, 1, 1>
{
  Point3DConsistencySameCamError(const Eigen::Vector3d &point3d_1,
                                 const Eigen::Vector3d &point3d_2)
      : point3d_1_(point3d_1), point3d_2_(point3d_2) {}

  template <typename T>
  bool operator()(const T *center,
                  const T *scale_1,
                  const T *scale_2,
                  T *residuals) const
  {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
    residuals_vec =
        (point3d_1_.cast<T>() / scale_1[0] +
         Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center)) -
        (point3d_2_.cast<T>() / scale_2[0] +
         Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center));
    return true;
  }

private:
  const Eigen::Vector3d point3d_1_;
  const Eigen::Vector3d point3d_2_;
};

// ----------------------------------------
// DepthConsistencySameCamError
// ----------------------------------------
// Cost function that enforces depth consistency between two observations
// with different scales: log(depth_1/scale_1) = log(depth_2/scale_2)
struct DepthConsistencySameCamError
    : public colmap::AutoDiffCostFunctor<DepthConsistencySameCamError, 1, 1, 1>
{
  DepthConsistencySameCamError(const double depth_1,
                               const double depth_2)
      : depth_1_(depth_1), depth_2_(depth_2) {}

  template <typename T>
  bool operator()(const T *scale_1,
                  const T *scale_2,
                  T *residuals) const
  {
    residuals[0] = ceres::log(T(depth_1_) / scale_1[0]) - ceres::log(T(depth_2_) / scale_2[0]);
    return true;
  }

private:
  const double depth_1_;
  const double depth_2_;
};
