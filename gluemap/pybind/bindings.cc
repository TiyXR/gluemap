#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cost_functions.h"

#include <algorithm>
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

// Solve a Ceres problem with CUDA GPU acceleration
void SolveCUDA(const ceres::Solver::Options &input_options,
               ceres::Problem *problem, ceres::Solver::Summary *summary) {
  ceres::Solver::Options options = input_options;

#ifdef CERES_HAS_CUDA
  switch (options.linear_solver_type) {
  case ceres::SPARSE_NORMAL_CHOLESKY:
  case ceres::SPARSE_SCHUR:
#ifndef CERES_NO_CUDSS
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
#endif
    break;
  case ceres::DENSE_NORMAL_CHOLESKY:
  case ceres::DENSE_SCHUR:
  case ceres::DENSE_QR:
    options.dense_linear_algebra_library_type = ceres::CUDA;
    break;
  default:
    break;
  }
#endif

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

PYBIND11_MODULE(pygluemap, m) {
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

  m.def("is_cuda_available", &IsCUDAAvailable,
        "Returns True if the module was compiled with CUDA support.");

  m.def("is_cuda_sparse_available", &IsCUDSSAvailable,
        "Returns True if the module was compiled with CUDA sparse/cuDSS "
        "support.");

  m.def("evaluate_connected_crs", &EvaluateConnectedCRS,
        py::arg("problem"), py::arg("ordered_parameter_addresses"),
        py::arg("seed_parameter_addresses"),
        py::arg("apply_loss_function") = true, py::arg("num_threads") = 1,
        "Evaluate only residual blocks connected to seed parameter blocks, "
        "while preserving the requested tangent-column ordering.");

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
