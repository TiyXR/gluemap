#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cost_functions.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ceres/ceres.h>

namespace py = pybind11;
using namespace pybind11::literals;

// Forward declarations from track_selection.cc
py::tuple ComputeTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<int64_t, int> &sift_count,
    int min_num_support_abs);

py::tuple ComputeVirtualTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<uint64_t, int> &pair_count_in,
    int min_num_support_abs);

py::array_t<int64_t> ComputeConnectedComponentsWrapper(
    int64_t node_count,
    py::array_t<int64_t, py::array::c_style> edge_first,
    py::array_t<int64_t, py::array::c_style> edge_second);

py::array_t<int64_t> BatchSpatialInternWrapper(
    py::array_t<int64_t, py::array::c_style> existing_frames,
    py::array_t<double, py::array::c_style> existing_x,
    py::array_t<double, py::array::c_style> existing_y,
    const std::vector<std::string> &existing_uids,
    py::array_t<int64_t, py::array::c_style> incoming_frames,
    py::array_t<double, py::array::c_style> incoming_x,
    py::array_t<double, py::array::c_style> incoming_y,
    const std::vector<std::string> &incoming_uids, double radius);

std::vector<std::string> BatchObservationUidsWrapper(
    const std::string &prediction_uid,
    py::array_t<int64_t, py::array::c_style> track_indexes,
    py::array_t<int64_t, py::array::c_style> view_indexes,
    const std::vector<std::string> &frame_uids);

void BindActiveTrackGraph(py::module_ &module);

// Helper function to create a ProductManifold for 7D pose (quat + trans)
ceres::Manifold *CreatePoseManifold() {
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::EuclideanManifold<3>>();
}

// Helper function to create a ProductManifold for 7D pose with fixed
// translation component
ceres::Manifold *
CreatePoseManifoldWithFixedTransComponent(int fixed_component) {
  std::vector<int> constant_indices = {fixed_component};
  return new ceres::ProductManifold<ceres::EigenQuaternionManifold,
                                    ceres::SubsetManifold>(
      ceres::EigenQuaternionManifold(),
      ceres::SubsetManifold(3, constant_indices));
}

// Helper function to create a manifold that fixes rotation but allows
// translation
ceres::Manifold *CreateTranslationOnlyManifold() {
  // Fix all 4 quaternion components, allow 3 translation components
  std::vector<int> constant_quat = {0, 1, 2, 3};
  return new ceres::ProductManifold<ceres::SubsetManifold,
                                    ceres::EuclideanManifold<3>>(
      ceres::SubsetManifold(4, constant_quat), ceres::EuclideanManifold<3>());
}

void ConfigureCUDA(ceres::Solver::Options *options) {
#ifdef CERES_HAS_CUDA
  switch (options->linear_solver_type) {
  case ceres::SPARSE_NORMAL_CHOLESKY:
  case ceres::SPARSE_SCHUR:
#ifndef CERES_NO_CUDSS
    options->sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
#endif
    break;
  case ceres::DENSE_NORMAL_CHOLESKY:
  case ceres::DENSE_SCHUR:
  case ceres::DENSE_QR:
    options->dense_linear_algebra_library_type = ceres::CUDA;
    break;
  default:
    break;
  }
#endif
}

// Solve a Ceres problem with CUDA GPU acceleration
void SolveCUDA(const ceres::Solver::Options &input_options,
               ceres::Problem *problem, ceres::Solver::Summary *summary) {
  ceres::Solver::Options options = input_options;
  ConfigureCUDA(&options);

  py::gil_scoped_release release;
  ceres::Solve(options, problem, summary);
}

void SolveWithBAOrdering(
    const ceres::Solver::Options &input_options, ceres::Problem *problem,
    ceres::Solver::Summary *summary,
    py::array_t<uint64_t, py::array::c_style> point_addresses,
    py::array_t<uint64_t, py::array::c_style> pose_addresses,
    uint64_t camera_address, bool use_cuda) {
  if (problem == nullptr || camera_address == 0) {
    throw std::invalid_argument("ordered BA problem is invalid");
  }
  const py::buffer_info points = point_addresses.request();
  const py::buffer_info poses = pose_addresses.request();
  if (points.ndim != 1 || poses.ndim != 1 || points.shape[0] < 1 ||
      poses.shape[0] < 1) {
    throw std::invalid_argument("ordered BA dimensions are invalid");
  }
  auto *point_values = static_cast<const uint64_t *>(points.ptr);
  auto *pose_values = static_cast<const uint64_t *>(poses.ptr);
  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  std::unordered_set<double *> ordered_blocks;
  ordered_blocks.reserve(static_cast<size_t>(points.shape[0] +
                                             poses.shape[0] + 1));
  for (py::ssize_t index = 0; index < points.shape[0]; ++index) {
    auto *parameter = reinterpret_cast<double *>(
        static_cast<uintptr_t>(point_values[index]));
    if (!problem->HasParameterBlock(parameter) ||
        !ordered_blocks.insert(parameter).second) {
      throw std::invalid_argument("ordered BA point block is invalid");
    }
    ordering->AddElementToGroup(parameter, 0);
  }
  for (py::ssize_t index = 0; index < poses.shape[0]; ++index) {
    auto *parameter = reinterpret_cast<double *>(
        static_cast<uintptr_t>(pose_values[index]));
    if (!problem->HasParameterBlock(parameter) ||
        !ordered_blocks.insert(parameter).second) {
      throw std::invalid_argument("ordered BA pose block is invalid");
    }
    ordering->AddElementToGroup(parameter, 1);
  }
  auto *camera = reinterpret_cast<double *>(
      static_cast<uintptr_t>(camera_address));
  if (!problem->HasParameterBlock(camera) ||
      !ordered_blocks.insert(camera).second) {
    throw std::invalid_argument("ordered BA camera block is invalid");
  }
  ordering->AddElementToGroup(camera, 1);

  ceres::Solver::Options options = input_options;
  options.linear_solver_ordering = std::move(ordering);
  if (use_cuda) {
    ConfigureCUDA(&options);
  }
  py::gil_scoped_release release;
  ceres::Solve(options, problem, summary);
}

bool IsCUDAAvailable() {
#ifdef CERES_HAS_CUDA
  return true;
#else
  return false;
#endif
}

bool IsCUDSSAvailable() {
#if defined(CERES_HAS_CUDA) && !defined(CERES_NO_CUDSS)
  return true;
#else
  return false;
#endif
}

class ReprojectionResidualBatch {
public:
  ReprojectionResidualBatch(
      ceres::Problem *problem,
      std::vector<ceres::ResidualBlockId> residual_blocks)
      : problem_(problem), residual_blocks_(std::move(residual_blocks)) {}

  size_t Size() const { return residual_blocks_.size(); }

  size_t ActiveSize() const {
    return static_cast<size_t>(std::count_if(
        residual_blocks_.begin(), residual_blocks_.end(),
        [](ceres::ResidualBlockId value) { return value != nullptr; }));
  }

  void RemoveIndices(
      py::array_t<int64_t, py::array::c_style> batch_indices) {
    if (problem_ == nullptr) {
      throw std::invalid_argument("reprojection batch is detached");
    }
    const py::buffer_info indices = batch_indices.request();
    if (indices.ndim != 1) {
      throw std::invalid_argument("reprojection batch indices are invalid");
    }
    auto *values = static_cast<const int64_t *>(indices.ptr);
    std::unordered_set<size_t> unique_indices;
    unique_indices.reserve(static_cast<size_t>(indices.shape[0]));
    for (py::ssize_t offset = 0; offset < indices.shape[0]; ++offset) {
      if (values[offset] < 0 ||
          static_cast<size_t>(values[offset]) >= residual_blocks_.size() ||
          residual_blocks_[static_cast<size_t>(values[offset])] == nullptr ||
          !unique_indices.insert(static_cast<size_t>(values[offset])).second) {
        throw std::invalid_argument(
            "reprojection batch index is absent");
      }
    }
    py::gil_scoped_release release;
    for (py::ssize_t offset = 0; offset < indices.shape[0]; ++offset) {
      const size_t index = static_cast<size_t>(values[offset]);
      problem_->RemoveResidualBlock(residual_blocks_[index]);
      residual_blocks_[index] = nullptr;
    }
  }

  void Remove() {
    if (problem_ == nullptr) {
      return;
    }
    py::gil_scoped_release release;
    for (ceres::ResidualBlockId &residual : residual_blocks_) {
      if (residual != nullptr) {
        problem_->RemoveResidualBlock(residual);
        residual = nullptr;
      }
    }
    problem_ = nullptr;
  }

private:
  ceres::Problem *problem_;
  std::vector<ceres::ResidualBlockId> residual_blocks_;
};

std::unique_ptr<ReprojectionResidualBatch> AddReprojectionResidualBatch(
    ceres::Problem *problem, int camera_model_id,
    py::array_t<uint64_t, py::array::c_style> point_addresses,
    py::array_t<uint64_t, py::array::c_style> pose_addresses,
    uint64_t camera_address,
    py::array_t<double, py::array::c_style> observation_xy,
    ceres::LossFunction *loss_function) {
  if (problem == nullptr || camera_address == 0) {
    throw std::invalid_argument("reprojection batch problem is invalid");
  }
  const py::buffer_info points = point_addresses.request();
  const py::buffer_info poses = pose_addresses.request();
  const py::buffer_info xy = observation_xy.request();
  if (points.ndim != 1 || poses.ndim != 1 || xy.ndim != 2 ||
      xy.shape[1] != 2 || points.shape[0] != poses.shape[0] ||
      points.shape[0] != xy.shape[0] || points.shape[0] < 1) {
    throw std::invalid_argument("reprojection batch dimensions differ");
  }
  auto *point_values = static_cast<const uint64_t *>(points.ptr);
  auto *pose_values = static_cast<const uint64_t *>(poses.ptr);
  auto *measurements = static_cast<const double *>(xy.ptr);
  auto *camera = reinterpret_cast<double *>(
      static_cast<uintptr_t>(camera_address));
  if (!problem->HasParameterBlock(camera)) {
    throw std::invalid_argument("reprojection batch camera is absent");
  }

  std::vector<ceres::ResidualBlockId> residuals;
  residuals.reserve(static_cast<size_t>(points.shape[0]));
  {
    py::gil_scoped_release release;
    for (py::ssize_t index = 0; index < points.shape[0]; ++index) {
      auto *point = reinterpret_cast<double *>(
          static_cast<uintptr_t>(point_values[index]));
      auto *pose = reinterpret_cast<double *>(
          static_cast<uintptr_t>(pose_values[index]));
      if (!problem->HasParameterBlock(point) ||
          !problem->HasParameterBlock(pose)) {
        throw std::invalid_argument(
            "reprojection batch parameter is absent");
      }
      const Eigen::Vector2d point2d(measurements[index * 2],
                                    measurements[index * 2 + 1]);
      ceres::CostFunction *cost =
          colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
              static_cast<colmap::CameraModelId>(camera_model_id), point2d);
      residuals.push_back(problem->AddResidualBlock(
          cost, loss_function, point, pose, camera));
    }
  }
  return std::make_unique<ReprojectionResidualBatch>(problem,
                                                      std::move(residuals));
}

std::unique_ptr<ReprojectionResidualBatch>
AddReprojectionResidualBatchImplicitParameters(
    ceres::Problem *problem, int camera_model_id,
    py::array_t<uint64_t, py::array::c_style> point_addresses,
    py::array_t<uint64_t, py::array::c_style> pose_addresses,
    py::array_t<uint8_t, py::array::c_style> fixed_pose_flags,
    uint64_t camera_address,
    py::array_t<double, py::array::c_style> observation_xy,
    ceres::LossFunction *loss_function) {
  if (problem == nullptr || camera_address == 0) {
    throw std::invalid_argument("implicit reprojection batch problem is invalid");
  }
  const py::buffer_info points = point_addresses.request();
  const py::buffer_info poses = pose_addresses.request();
  const py::buffer_info fixed = fixed_pose_flags.request();
  const py::buffer_info xy = observation_xy.request();
  if (points.ndim != 1 || poses.ndim != 1 || fixed.ndim != 1 ||
      xy.ndim != 2 ||
      xy.shape[1] != 2 || points.shape[0] != poses.shape[0] ||
      points.shape[0] != fixed.shape[0] || points.shape[0] != xy.shape[0] ||
      points.shape[0] < 1) {
    throw std::invalid_argument(
        "implicit reprojection batch dimensions differ");
  }
  auto *point_values = static_cast<const uint64_t *>(points.ptr);
  auto *pose_values = static_cast<const uint64_t *>(poses.ptr);
  auto *fixed_values = static_cast<const uint8_t *>(fixed.ptr);
  auto *measurements = static_cast<const double *>(xy.ptr);
  auto *camera = reinterpret_cast<double *>(
      static_cast<uintptr_t>(camera_address));

  std::vector<ceres::ResidualBlockId> residuals;
  residuals.reserve(static_cast<size_t>(points.shape[0]));
  {
    py::gil_scoped_release release;
    for (py::ssize_t index = 0; index < points.shape[0]; ++index) {
      auto *point = reinterpret_cast<double *>(
          static_cast<uintptr_t>(point_values[index]));
      auto *pose = reinterpret_cast<double *>(
          static_cast<uintptr_t>(pose_values[index]));
      const Eigen::Vector2d point2d(measurements[index * 2],
                                    measurements[index * 2 + 1]);
      if (fixed_values[index] != 0) {
        colmap::Rigid3d cam_from_world;
        cam_from_world.params = Eigen::Map<const Eigen::Vector7d>(pose);
        ceres::CostFunction *cost = colmap::CreateCameraCostFunction<
            colmap::ReprojErrorConstantPoseCostFunctor>(
            static_cast<colmap::CameraModelId>(camera_model_id), point2d,
            cam_from_world);
        residuals.push_back(
            problem->AddResidualBlock(cost, loss_function, point, camera));
      } else {
        ceres::CostFunction *cost =
            colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
                static_cast<colmap::CameraModelId>(camera_model_id), point2d);
        // Match COLMAP DefaultBundleAdjuster::AddImageWithTrivialFrame: the
        // first variable-pose residual implicitly creates parameter blocks in
        // point, pose, camera order. Manifolds are assigned after all images.
        residuals.push_back(problem->AddResidualBlock(
            cost, loss_function, point, pose, camera));
      }
    }
  }
  return std::make_unique<ReprojectionResidualBatch>(problem,
                                                      std::move(residuals));
}

std::unique_ptr<ReprojectionResidualBatch>
AddReprojectionResidualCSRImplicitParameters(
    ceres::Problem *problem, int camera_model_id,
    py::array_t<uint64_t, py::array::c_style> point_addresses,
    py::array_t<uint64_t, py::array::c_style> pose_addresses,
    py::array_t<uint8_t, py::array::c_style> fixed_pose_flags,
    py::array_t<uint64_t, py::array::c_style> track_offsets,
    py::array_t<int64_t, py::array::c_style> observation_frame_indices,
    uint64_t camera_address,
    py::array_t<double, py::array::c_style> observation_xy,
    ceres::LossFunction *loss_function) {
  if (problem == nullptr || camera_address == 0) {
    throw std::invalid_argument("CSR reprojection batch problem is invalid");
  }
  const py::buffer_info points = point_addresses.request();
  const py::buffer_info poses = pose_addresses.request();
  const py::buffer_info fixed = fixed_pose_flags.request();
  const py::buffer_info offsets = track_offsets.request();
  const py::buffer_info frames = observation_frame_indices.request();
  const py::buffer_info xy = observation_xy.request();
  if (points.ndim != 1 || poses.ndim != 1 || fixed.ndim != 1 ||
      offsets.ndim != 1 || frames.ndim != 1 || xy.ndim != 2 ||
      xy.shape[1] != 2 || points.shape[0] < 1 || poses.shape[0] < 1 ||
      fixed.shape[0] != poses.shape[0] ||
      offsets.shape[0] != points.shape[0] + 1 ||
      frames.shape[0] != xy.shape[0] || frames.shape[0] < 1) {
    throw std::invalid_argument("CSR reprojection batch dimensions differ");
  }

  auto *point_values = static_cast<const uint64_t *>(points.ptr);
  auto *pose_values = static_cast<const uint64_t *>(poses.ptr);
  auto *fixed_values = static_cast<const uint8_t *>(fixed.ptr);
  auto *offset_values = static_cast<const uint64_t *>(offsets.ptr);
  auto *frame_values = static_cast<const int64_t *>(frames.ptr);
  auto *measurements = static_cast<const double *>(xy.ptr);
  auto *camera = reinterpret_cast<double *>(
      static_cast<uintptr_t>(camera_address));
  const size_t point_count = static_cast<size_t>(points.shape[0]);
  const size_t pose_count = static_cast<size_t>(poses.shape[0]);
  const size_t observation_count = static_cast<size_t>(frames.shape[0]);

  std::vector<ceres::ResidualBlockId> residuals;
  residuals.reserve(observation_count);
  {
    py::gil_scoped_release release;
    if (offset_values[0] != 0 ||
        offset_values[point_count] != observation_count) {
      throw std::invalid_argument("CSR reprojection offsets are invalid");
    }

    // Stable counting sort converts compact track-major CSR input into the
    // image-major residual insertion order used by COLMAP. This preserves
    // Ceres parameter ordering without Python tuple materialization/sorting.
    std::vector<size_t> frame_counts(pose_count, 0);
    std::vector<size_t> observation_tracks(observation_count, 0);
    for (size_t track = 0; track < point_count; ++track) {
      const uint64_t begin = offset_values[track];
      const uint64_t end = offset_values[track + 1];
      if (begin > end || end > observation_count) {
        throw std::invalid_argument("CSR reprojection offsets are not monotonic");
      }
      for (uint64_t observation = begin; observation < end; ++observation) {
        const int64_t frame = frame_values[observation];
        if (frame < 0 || static_cast<size_t>(frame) >= pose_count) {
          throw std::invalid_argument("CSR reprojection frame is invalid");
        }
        ++frame_counts[static_cast<size_t>(frame)];
        observation_tracks[static_cast<size_t>(observation)] = track;
      }
    }

    std::vector<size_t> frame_cursor(pose_count, 0);
    size_t offset = 0;
    for (size_t frame = 0; frame < pose_count; ++frame) {
      frame_cursor[frame] = offset;
      offset += frame_counts[frame];
    }
    std::vector<size_t> image_major_observations(observation_count, 0);
    for (size_t observation = 0; observation < observation_count;
         ++observation) {
      const size_t frame = static_cast<size_t>(frame_values[observation]);
      image_major_observations[frame_cursor[frame]++] = observation;
    }

    for (const size_t observation : image_major_observations) {
      const size_t track = observation_tracks[observation];
      const size_t frame = static_cast<size_t>(frame_values[observation]);
      auto *point = reinterpret_cast<double *>(
          static_cast<uintptr_t>(point_values[track]));
      auto *pose = reinterpret_cast<double *>(
          static_cast<uintptr_t>(pose_values[frame]));
      const Eigen::Vector2d point2d(measurements[observation * 2],
                                    measurements[observation * 2 + 1]);
      if (fixed_values[frame] != 0) {
        colmap::Rigid3d cam_from_world;
        cam_from_world.params = Eigen::Map<const Eigen::Vector7d>(pose);
        ceres::CostFunction *cost = colmap::CreateCameraCostFunction<
            colmap::ReprojErrorConstantPoseCostFunctor>(
            static_cast<colmap::CameraModelId>(camera_model_id), point2d,
            cam_from_world);
        residuals.push_back(
            problem->AddResidualBlock(cost, loss_function, point, camera));
      } else {
        ceres::CostFunction *cost =
            colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(
                static_cast<colmap::CameraModelId>(camera_model_id), point2d);
        residuals.push_back(problem->AddResidualBlock(
            cost, loss_function, point, pose, camera));
      }
    }
  }
  return std::make_unique<ReprojectionResidualBatch>(problem,
                                                      std::move(residuals));
}

py::dict EvaluateConnectedCRS(
    ceres::Problem *problem,
    const std::vector<uintptr_t> &ordered_parameter_addresses,
    const std::vector<uintptr_t> &seed_parameter_addresses,
    bool apply_loss_function, int num_threads) {
  if (problem == nullptr || ordered_parameter_addresses.empty() ||
      seed_parameter_addresses.empty()) {
    throw std::invalid_argument(
        "connected CRS evaluation requires problem, parameters and seeds");
  }
  if (num_threads < 1) {
    throw std::invalid_argument("connected CRS thread count must be positive");
  }

  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = apply_loss_function;
  options.num_threads = num_threads;
  options.parameter_blocks.reserve(ordered_parameter_addresses.size());
  for (const uintptr_t address : ordered_parameter_addresses) {
    auto *parameter = reinterpret_cast<double *>(address);
    if (!problem->HasParameterBlock(parameter)) {
      throw std::invalid_argument(
          "ordered parameter address is absent from Ceres problem");
    }
    options.parameter_blocks.push_back(parameter);
  }

  std::unordered_set<double *> seed_parameters;
  for (const uintptr_t address : seed_parameter_addresses) {
    auto *parameter = reinterpret_cast<double *>(address);
    if (!problem->HasParameterBlock(parameter)) {
      throw std::invalid_argument(
          "seed parameter address is absent from Ceres problem");
    }
    seed_parameters.insert(parameter);
  }

  std::vector<ceres::ResidualBlockId> all_residuals;
  problem->GetResidualBlocks(&all_residuals);
  options.residual_blocks.reserve(all_residuals.size());
  std::vector<double *> residual_parameters;
  for (const ceres::ResidualBlockId residual : all_residuals) {
    residual_parameters.clear();
    problem->GetParameterBlocksForResidualBlock(residual,
                                                &residual_parameters);
    if (std::any_of(residual_parameters.begin(), residual_parameters.end(),
                    [&seed_parameters](double *parameter) {
                      return seed_parameters.count(parameter) != 0;
                    })) {
      options.residual_blocks.push_back(residual);
    }
  }
  if (options.residual_blocks.empty()) {
    throw std::invalid_argument("seed parameters have no connected residuals");
  }

  double cost = 0.0;
  std::vector<double> residuals;
  ceres::CRSMatrix jacobian;
  bool evaluated = false;
  {
    py::gil_scoped_release release;
    evaluated = problem->Evaluate(options, &cost, &residuals, nullptr,
                                  &jacobian);
  }
  if (!evaluated) {
    throw std::runtime_error("connected Ceres CRS evaluation failed");
  }

  auto residual_array = py::array_t<double>(residuals.size());
  std::copy(residuals.begin(), residuals.end(), residual_array.mutable_data());
  auto row_array = py::array_t<int64_t>(jacobian.rows.size());
  std::copy(jacobian.rows.begin(), jacobian.rows.end(),
            row_array.mutable_data());
  auto column_array = py::array_t<int64_t>(jacobian.cols.size());
  std::copy(jacobian.cols.begin(), jacobian.cols.end(),
            column_array.mutable_data());
  auto value_array = py::array_t<double>(jacobian.values.size());
  std::copy(jacobian.values.begin(), jacobian.values.end(),
            value_array.mutable_data());

  py::dict result;
  result["cost"] = cost;
  result["residuals"] = std::move(residual_array);
  result["rowOffsets"] = std::move(row_array);
  result["columnIndices"] = std::move(column_array);
  result["jacobianValues"] = std::move(value_array);
  result["residualBlockCount"] = options.residual_blocks.size();
  result["residualCount"] = residuals.size();
  result["columnCount"] = jacobian.num_cols;
  return result;
}

struct ConnectedNormalContribution {
  int camera_index = -1;
  std::array<double, 36> camera_hessian{};
  std::array<double, 6> camera_gradient{};
  std::array<double, 18> camera_point_hessian{};
};

py::dict EvaluateConnectedNormalBlocks(
    ceres::Problem *problem,
    const std::vector<uintptr_t> &ordered_parameter_addresses,
    const std::vector<uintptr_t> &seed_parameter_addresses,
    int camera_parameter_count, int point_parameter_count,
    bool apply_loss_function, int num_threads) {
  if (problem == nullptr || camera_parameter_count < 1 ||
      point_parameter_count < 1 || num_threads < 1 ||
      ordered_parameter_addresses.size() !=
          static_cast<size_t>(camera_parameter_count +
                              point_parameter_count) ||
      seed_parameter_addresses.empty()) {
    throw std::invalid_argument(
        "connected normal block evaluation identity is invalid");
  }

  const auto selection_started = std::chrono::steady_clock::now();
  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = apply_loss_function;
  options.num_threads = num_threads;
  options.parameter_blocks.reserve(ordered_parameter_addresses.size());
  for (const uintptr_t address : ordered_parameter_addresses) {
    auto *parameter = reinterpret_cast<double *>(address);
    if (!problem->HasParameterBlock(parameter)) {
      throw std::invalid_argument(
          "ordered parameter address is absent from Ceres problem");
    }
    options.parameter_blocks.push_back(parameter);
  }

  std::unordered_set<double *> seed_parameters;
  seed_parameters.reserve(seed_parameter_addresses.size());
  for (const uintptr_t address : seed_parameter_addresses) {
    auto *parameter = reinterpret_cast<double *>(address);
    if (!problem->HasParameterBlock(parameter)) {
      throw std::invalid_argument(
          "seed parameter address is absent from Ceres problem");
    }
    seed_parameters.insert(parameter);
  }

  std::vector<ceres::ResidualBlockId> all_residuals;
  problem->GetResidualBlocks(&all_residuals);
  options.residual_blocks.reserve(all_residuals.size());
  std::vector<double *> residual_parameters;
  for (const ceres::ResidualBlockId residual : all_residuals) {
    residual_parameters.clear();
    problem->GetParameterBlocksForResidualBlock(residual,
                                                &residual_parameters);
    if (std::any_of(residual_parameters.begin(), residual_parameters.end(),
                    [&seed_parameters](double *parameter) {
                      return seed_parameters.count(parameter) != 0;
                    })) {
      options.residual_blocks.push_back(residual);
    }
  }
  if (options.residual_blocks.empty()) {
    throw std::invalid_argument("seed parameters have no connected residuals");
  }
  const auto selection_finished = std::chrono::steady_clock::now();

  double cost = 0.0;
  std::vector<double> residuals;
  ceres::CRSMatrix jacobian;
  bool evaluated = false;
  const auto evaluation_started = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release release;
    evaluated = problem->Evaluate(options, &cost, &residuals, nullptr,
                                  &jacobian);
  }
  const auto evaluation_finished = std::chrono::steady_clock::now();
  if (!evaluated) {
    throw std::runtime_error("connected Ceres normal evaluation failed");
  }

  const int camera_dimension = camera_parameter_count * 6;
  const int point_dimension = point_parameter_count * 3;
  const int expected_columns = camera_dimension + point_dimension;
  const size_t residual_count = residuals.size();
  if (jacobian.num_cols != expected_columns ||
      jacobian.num_rows != static_cast<int>(residual_count) ||
      jacobian.rows.size() != residual_count + 1 || jacobian.rows.front() != 0 ||
      jacobian.rows.back() != static_cast<int>(jacobian.values.size()) ||
      jacobian.cols.size() != jacobian.values.size()) {
    throw std::runtime_error("connected Ceres normal CRS layout is invalid");
  }

  const auto normal_started = std::chrono::steady_clock::now();
  std::vector<int> row_points(residual_count, -1);
  std::vector<int> row_cameras(residual_count, -1);
  std::vector<size_t> point_row_counts(point_parameter_count, 0);
  for (size_t row = 0; row < residual_count; ++row) {
    int point_index = -1;
    int camera_index = -1;
    for (int offset = jacobian.rows[row]; offset < jacobian.rows[row + 1];
         ++offset) {
      const int column = jacobian.cols[offset];
      if (column < 0 || column >= expected_columns) {
        throw std::runtime_error("connected Ceres normal column is invalid");
      }
      if (column < camera_dimension) {
        const int candidate = column / 6;
        if (camera_index >= 0 && camera_index != candidate) {
          throw std::runtime_error(
              "connected Ceres row contains multiple camera blocks");
        }
        camera_index = candidate;
      } else {
        const int candidate = (column - camera_dimension) / 3;
        if (point_index >= 0 && point_index != candidate) {
          throw std::runtime_error(
              "connected Ceres row contains multiple point blocks");
        }
        point_index = candidate;
      }
    }
    if (point_index < 0 || point_index >= point_parameter_count) {
      throw std::runtime_error(
          "connected Ceres row has no ordered point block");
    }
    row_points[row] = point_index;
    row_cameras[row] = camera_index;
    ++point_row_counts[point_index];
  }

  std::vector<size_t> point_row_offsets(point_parameter_count + 1, 0);
  std::partial_sum(point_row_counts.begin(), point_row_counts.end(),
                   point_row_offsets.begin() + 1);
  std::vector<size_t> point_row_cursor = point_row_offsets;
  std::vector<size_t> point_rows(residual_count, 0);
  for (size_t row = 0; row < residual_count; ++row) {
    const int point_index = row_points[row];
    point_rows[point_row_cursor[point_index]++] = row;
  }

  std::vector<double> point_hessian(
      static_cast<size_t>(point_parameter_count) * 9, 0.0);
  std::vector<double> point_gradient(
      static_cast<size_t>(point_parameter_count) * 3, 0.0);
  std::vector<std::vector<ConnectedNormalContribution>> point_contributions(
      point_parameter_count);

#pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int point_index = 0; point_index < point_parameter_count;
       ++point_index) {
    std::unordered_map<int, ConnectedNormalContribution> by_camera;
    for (size_t cursor = point_row_offsets[point_index];
         cursor < point_row_offsets[point_index + 1]; ++cursor) {
      const size_t row = point_rows[cursor];
      const int camera_index = row_cameras[row];
      std::array<double, 6> camera_jacobian{};
      std::array<double, 3> point_jacobian{};
      for (int offset = jacobian.rows[row]; offset < jacobian.rows[row + 1];
           ++offset) {
        const int column = jacobian.cols[offset];
        const double value = jacobian.values[offset];
        if (column < camera_dimension) {
          camera_jacobian[column % 6] = value;
        } else {
          point_jacobian[(column - camera_dimension) % 3] = value;
        }
      }
      const double residual = residuals[row];
      double *point_hessian_block =
          point_hessian.data() + static_cast<size_t>(point_index) * 9;
      double *point_gradient_block =
          point_gradient.data() + static_cast<size_t>(point_index) * 3;
      for (int point_row = 0; point_row < 3; ++point_row) {
        point_gradient_block[point_row] +=
            point_jacobian[point_row] * residual;
        for (int point_column = 0; point_column < 3; ++point_column) {
          point_hessian_block[point_row * 3 + point_column] +=
              point_jacobian[point_row] * point_jacobian[point_column];
        }
      }
      if (camera_index < 0) {
        continue;
      }
      auto &contribution = by_camera[camera_index];
      contribution.camera_index = camera_index;
      for (int camera_row = 0; camera_row < 6; ++camera_row) {
        contribution.camera_gradient[camera_row] +=
            camera_jacobian[camera_row] * residual;
        for (int camera_column = 0; camera_column < 6; ++camera_column) {
          contribution.camera_hessian[camera_row * 6 + camera_column] +=
              camera_jacobian[camera_row] * camera_jacobian[camera_column];
        }
        for (int point_column = 0; point_column < 3; ++point_column) {
          contribution.camera_point_hessian[camera_row * 3 + point_column] +=
              camera_jacobian[camera_row] * point_jacobian[point_column];
        }
      }
    }
    auto &ordered = point_contributions[point_index];
    ordered.reserve(by_camera.size());
    for (auto &entry : by_camera) {
      ordered.push_back(std::move(entry.second));
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const ConnectedNormalContribution &left,
                 const ConnectedNormalContribution &right) {
                return left.camera_index < right.camera_index;
              });
  }

  size_t camera_point_block_count = 0;
  for (const auto &value : point_contributions) {
    camera_point_block_count += value.size();
  }
  std::vector<double> camera_hessian(
      static_cast<size_t>(camera_dimension) * camera_dimension, 0.0);
  std::vector<double> camera_gradient(camera_dimension, 0.0);
  std::vector<int64_t> block_point_indexes;
  std::vector<int64_t> block_camera_indexes;
  std::vector<double> camera_point_hessian;
  block_point_indexes.reserve(camera_point_block_count);
  block_camera_indexes.reserve(camera_point_block_count);
  camera_point_hessian.reserve(camera_point_block_count * 18);
  for (int point_index = 0; point_index < point_parameter_count;
       ++point_index) {
    for (const auto &contribution : point_contributions[point_index]) {
      const int camera_index = contribution.camera_index;
      for (int row = 0; row < 6; ++row) {
        camera_gradient[camera_index * 6 + row] +=
            contribution.camera_gradient[row];
        for (int column = 0; column < 6; ++column) {
          camera_hessian[static_cast<size_t>(camera_index * 6 + row) *
                             camera_dimension +
                         camera_index * 6 + column] +=
              contribution.camera_hessian[row * 6 + column];
        }
      }
      block_point_indexes.push_back(point_index);
      block_camera_indexes.push_back(camera_index);
      camera_point_hessian.insert(camera_point_hessian.end(),
                                  contribution.camera_point_hessian.begin(),
                                  contribution.camera_point_hessian.end());
    }
  }
  const auto normal_finished = std::chrono::steady_clock::now();

  const auto seconds = [](const auto &start, const auto &finish) {
    return std::chrono::duration<double>(finish - start).count();
  };
  auto camera_hessian_array = py::array_t<double>(
      {static_cast<py::ssize_t>(camera_dimension),
       static_cast<py::ssize_t>(camera_dimension)});
  std::copy(camera_hessian.begin(), camera_hessian.end(),
            camera_hessian_array.mutable_data());
  auto camera_gradient_array =
      py::array_t<double>({static_cast<py::ssize_t>(camera_dimension)});
  std::copy(camera_gradient.begin(), camera_gradient.end(),
            camera_gradient_array.mutable_data());
  auto point_hessian_array = py::array_t<double>(
      {static_cast<py::ssize_t>(point_parameter_count),
       static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)});
  std::copy(point_hessian.begin(), point_hessian.end(),
            point_hessian_array.mutable_data());
  auto point_gradient_array = py::array_t<double>(
      {static_cast<py::ssize_t>(point_parameter_count),
       static_cast<py::ssize_t>(3)});
  std::copy(point_gradient.begin(), point_gradient.end(),
            point_gradient_array.mutable_data());
  auto block_point_array = py::array_t<int64_t>(
      {static_cast<py::ssize_t>(camera_point_block_count)});
  std::copy(block_point_indexes.begin(), block_point_indexes.end(),
            block_point_array.mutable_data());
  auto block_camera_array = py::array_t<int64_t>(
      {static_cast<py::ssize_t>(camera_point_block_count)});
  std::copy(block_camera_indexes.begin(), block_camera_indexes.end(),
            block_camera_array.mutable_data());
  auto camera_point_array = py::array_t<double>(
      {static_cast<py::ssize_t>(camera_point_block_count),
       static_cast<py::ssize_t>(6), static_cast<py::ssize_t>(3)});
  std::copy(camera_point_hessian.begin(), camera_point_hessian.end(),
            camera_point_array.mutable_data());

  py::dict result;
  result["cost"] = cost;
  result["cameraHessian"] = std::move(camera_hessian_array);
  result["cameraGradient"] = std::move(camera_gradient_array);
  result["pointHessian"] = std::move(point_hessian_array);
  result["pointGradient"] = std::move(point_gradient_array);
  result["blockPointIndexes"] = std::move(block_point_array);
  result["blockCameraIndexes"] = std::move(block_camera_array);
  result["cameraPointHessian"] = std::move(camera_point_array);
  result["residualBlockCount"] = options.residual_blocks.size();
  result["residualCount"] = residual_count;
  result["columnCount"] = jacobian.num_cols;
  result["jacobianNonzeroCount"] = jacobian.values.size();
  result["selectionWallSeconds"] =
      seconds(selection_started, selection_finished);
  result["evaluationWallSeconds"] =
      seconds(evaluation_started, evaluation_finished);
  result["normalBuildWallSeconds"] = seconds(normal_started, normal_finished);
  return result;
}

PYBIND11_MODULE(pygluemap, m) {
  BindActiveTrackGraph(m);
  py::module_::import("pyceres");

  m.def("RotationGeodesicError",
        &RotationGeodesicError::Create<const Eigen::Vector4d &>,
        py::arg("i_q_j"));

  m.def("PairwiseDirectionError",
        &PairwiseDirectionError::Create<const Eigen::Vector3d &>,
        py::arg("translation_obs"));

  m.def("CreateFejPosePriorCost", &CreateFejPosePriorCost,
        py::arg("factor"), py::arg("factor_residual"),
        py::arg("linearization"), py::return_value_policy::take_ownership,
        "Create a GIL-free dense Schur/FEJ pose prior cost function.");

  m.def("ReprojErrorCost",
        &colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor,
                                          const Eigen::Vector2d &>,
        "camera_model_id"_a, "point2D"_a, "Reprojection error.");

  m.def(
      "ReprojErrorCostWithNegativeDepth",
      &colmap::CreateCameraCostFunction<ReprojErrorCostWithNegativeDepthFunctor,
                                        const Eigen::Vector2d &>,
      "camera_model_id"_a, "point2D"_a,
      "Reprojection error with negative depth.");

  // Manifold creation helpers for 7D pose (quaternion + translation)
  m.def("CreatePoseManifold", &CreatePoseManifold,
        py::return_value_policy::take_ownership,
        "Create a ProductManifold for 7D pose (quaternion + translation).");

  m.def("CreatePoseManifoldWithFixedTransComponent",
        &CreatePoseManifoldWithFixedTransComponent, py::arg("fixed_component"),
        py::return_value_policy::take_ownership,
        "Create a ProductManifold for 7D pose with one translation component "
        "fixed.");

  m.def(
      "CreateTranslationOnlyManifold", &CreateTranslationOnlyManifold,
      py::return_value_policy::take_ownership,
      "Create a manifold that fixes rotation but allows translation to vary.");

  // CUDA GPU solver
  m.def("solve_cuda", &SolveCUDA, py::arg("options"), py::arg("problem"),
        py::arg("summary"),
        "Solve a Ceres problem with CUDA GPU acceleration.");

  m.def("solve_with_ba_ordering", &SolveWithBAOrdering,
        py::arg("options"), py::arg("problem"), py::arg("summary"),
        py::arg("point_addresses"), py::arg("pose_addresses"),
        py::arg("camera_address"), py::arg("use_cuda"),
        "Solve with explicit point-first BA elimination groups.");

  m.def("add_reprojection_residual_batch_implicit_parameters",
        &AddReprojectionResidualBatchImplicitParameters,
        py::arg("problem"), py::arg("camera_model_id"),
        py::arg("point_addresses"), py::arg("pose_addresses"),
        py::arg("fixed_pose_flags"),
        py::arg("camera_address"), py::arg("observation_xy"),
        py::arg("loss_function"),
        "Add image-major reprojection residuals while letting Ceres create "
        "parameter blocks in COLMAP order.");

  m.def("add_reprojection_residual_csr_implicit_parameters",
        &AddReprojectionResidualCSRImplicitParameters,
        py::arg("problem"), py::arg("camera_model_id"),
        py::arg("point_addresses"), py::arg("pose_addresses"),
        py::arg("fixed_pose_flags"), py::arg("track_offsets"),
        py::arg("observation_frame_indices"), py::arg("camera_address"),
        py::arg("observation_xy"), py::arg("loss_function"),
        "Add compact track-major CSR observations in deterministic "
        "image-major COLMAP parameter order.");

  m.def("is_cuda_available", &IsCUDAAvailable,
        "Returns True if the module was compiled with CUDA support.");

  m.def("is_cuda_sparse_available", &IsCUDSSAvailable,
        "Returns True if the module was compiled with CUDA sparse/cuDSS "
        "support.");

  py::class_<ReprojectionResidualBatch>(m, "ReprojectionResidualBatch")
      .def_property_readonly("size", &ReprojectionResidualBatch::Size)
      .def_property_readonly("active_size",
                             &ReprojectionResidualBatch::ActiveSize)
      .def("remove_indices", &ReprojectionResidualBatch::RemoveIndices,
           py::arg("batch_indices"),
           "Remove selected residuals in one GIL-free native call.")
      .def("remove", &ReprojectionResidualBatch::Remove,
           "Remove the complete native residual batch in one GIL-free call.");

  m.def("add_reprojection_residual_batch",
        &AddReprojectionResidualBatch, py::arg("problem"),
        py::arg("camera_model_id"), py::arg("point_addresses"),
        py::arg("pose_addresses"), py::arg("camera_address"),
        py::arg("observation_xy"), py::arg("loss_function"),
        "Add one deterministic visual residual batch without per-row Python "
        "calls.");

  m.def("evaluate_connected_crs", &EvaluateConnectedCRS,
        py::arg("problem"), py::arg("ordered_parameter_addresses"),
        py::arg("seed_parameter_addresses"),
        py::arg("apply_loss_function") = true, py::arg("num_threads") = 1,
        "Evaluate only residual blocks connected to seed parameter blocks, "
        "while preserving the requested tangent-column ordering.");

  m.def("evaluate_connected_normal_blocks", &EvaluateConnectedNormalBlocks,
        py::arg("problem"), py::arg("ordered_parameter_addresses"),
        py::arg("seed_parameter_addresses"),
        py::arg("camera_parameter_count"), py::arg("point_parameter_count"),
        py::arg("apply_loss_function") = true, py::arg("num_threads") = 1,
        "Evaluate connected residuals and build deterministic camera/point "
        "normal blocks with bounded OpenMP working memory.");

  // Numpy-based track selection: returns point3D IDs to delete.
  // Python then calls reconstruction.delete_point3d(id) for each.
  m.def("compute_tracks_to_delete", &ComputeTracksToDeleteWrapper,
        py::arg("point3d_ids"), py::arg("track_image_ids"),
        py::arg("track_pt2d_idxs"), py::arg("track_lengths"),
        py::arg("sift_count"), py::arg("min_num_support_abs") = 512,
        "Classify and select tracks. Returns (ids_to_delete, pair_count) where "
        "ids_to_delete is an int64 array of point3D IDs to delete and "
        "pair_count is a dict mapping (img_low, img_high) tuples to coverage "
        "counts after selection.");

  m.def("compute_virtual_tracks_to_delete",
        &ComputeVirtualTracksToDeleteWrapper, py::arg("point3d_ids"),
        py::arg("track_image_ids"), py::arg("track_pt2d_idxs"),
        py::arg("track_lengths"), py::arg("pair_count"),
        py::arg("min_num_support_abs") = 512,
        "Select virtual tracks given existing pair coverage. Returns "
        "(ids_to_delete, updated_pair_count). Tracks whose image pairs are "
        "all above min_num_support_abs are removed.");

  m.def("compute_connected_components", &ComputeConnectedComponentsWrapper,
        py::arg("node_count"), py::arg("edge_first"), py::arg("edge_second"),
        "Compute integer graph component labels with GIL-free OpenMP workers.");

  m.def("batch_spatial_intern", &BatchSpatialInternWrapper,
        py::arg("existing_frames"), py::arg("existing_x"),
        py::arg("existing_y"), py::arg("existing_uids"),
        py::arg("incoming_frames"), py::arg("incoming_x"),
        py::arg("incoming_y"), py::arg("incoming_uids"), py::arg("radius"),
        "Resolve one observation batch against a native per-frame spatial hash.");

  m.def("batch_observation_uids", &BatchObservationUidsWrapper,
        py::arg("prediction_uid"), py::arg("track_indexes"),
        py::arg("view_indexes"), py::arg("frame_uids"),
        "Compute stable observation SHA-256 identities with OpenMP workers.");
}
