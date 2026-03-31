#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.h"

#include <array>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ── Output struct ────────────────────────────────────────────────────────────
struct MergeResult {
    std::vector<std::array<double, 3>> merged_xyz;
    std::vector<int64_t>               merged_track_img_ids;
    std::vector<int32_t>               merged_track_pt2d_idxs;
    std::vector<int32_t>               merged_track_lens;
    std::unordered_map<int64_t, int>                         recon_offsets;
    std::unordered_map<int64_t, int64_t>                     tri_to_recon_id;
    std::unordered_map<int64_t, int>                         tri_offsets;
    std::unordered_map<int64_t, int>                         virtual_point_start;
    std::unordered_map<int64_t, std::unordered_set<int>>     negative_depth_observations;
};

// ── Core function: pure C++, no pybind11 ────────────────────────────────────
//
// Image data (parallel arrays, same order):
//   tri_names / tri_ids / tri_n_pts2d     – triangulated reconstruction
//   recon_names / recon_ids / recon_n_pts2d – base reconstruction
//
// 3D point data (CSR-like):
//   recon_xyz / recon_track_img_ids / recon_track_pt2d_idxs / recon_track_lens
//   tri_xyz   / tri_track_img_ids   / tri_track_pt2d_idxs   / tri_track_lens
//
// Returns MergeResult with merged arrays and updated state maps.
MergeResult MergeReconstructionData(
    const std::vector<std::string>&          tri_names,
    const std::vector<int64_t>&              tri_ids,
    const std::vector<int32_t>&              tri_n_pts2d,
    const std::vector<std::string>&          recon_names,
    const std::vector<int64_t>&              recon_ids,
    const std::vector<int32_t>&              recon_n_pts2d,
    const std::vector<std::array<double,3>>& recon_xyz,
    const std::vector<int64_t>&              recon_track_img_ids,
    const std::vector<int32_t>&              recon_track_pt2d_idxs,
    const std::vector<int32_t>&              recon_track_lens,
    const std::vector<std::array<double,3>>& tri_xyz,
    const std::vector<int64_t>&              tri_track_img_ids,
    const std::vector<int32_t>&              tri_track_pt2d_idxs,
    const std::vector<int32_t>&              tri_track_lens,
    std::unordered_map<int64_t, int>                     virtual_point_start,
    std::unordered_map<int64_t, std::unordered_set<int>> negative_depth_observations,
    bool triangulated_features_first) {

  const int64_t T = (int64_t)tri_ids.size();
  const int64_t R = (int64_t)recon_ids.size();
  const int64_t N = (int64_t)recon_xyz.size();
  const int64_t M = (int64_t)tri_xyz.size();

  // 1. Build name -> (id, n_pts2d) maps for both sides
  std::unordered_map<std::string, int64_t> tri_name_to_id;
  std::unordered_map<int64_t, int32_t>     tri_id_to_npts;
  for (int64_t i = 0; i < T; ++i) {
    tri_name_to_id[tri_names[i]] = tri_ids[i];
    tri_id_to_npts[tri_ids[i]]   = tri_n_pts2d[i];
  }

  std::unordered_map<std::string, int64_t> recon_name_to_id;
  std::unordered_map<int64_t, int32_t>     recon_id_to_npts;
  for (int64_t i = 0; i < R; ++i) {
    recon_name_to_id[recon_names[i]] = recon_ids[i];
    recon_id_to_npts[recon_ids[i]]   = recon_n_pts2d[i];
  }

  // 2. Build tri_id -> recon_id mapping via image names
  std::unordered_map<int64_t, int64_t> tri_to_recon_id;
  for (const auto& [name, tri_id] : tri_name_to_id) {
    const auto it = recon_name_to_id.find(name);
    if (it != recon_name_to_id.end())
      tri_to_recon_id[tri_id] = it->second;
  }

  // 3. Compute per-image offsets
  std::unordered_map<int64_t, int> recon_offset;
  std::unordered_map<int64_t, int> tri_offset;
  for (const auto& [tri_id, recon_id] : tri_to_recon_id) {
    const int n_tri   = tri_id_to_npts.count(tri_id)   ? tri_id_to_npts.at(tri_id)   : 0;
    const int n_recon = recon_id_to_npts.count(recon_id) ? recon_id_to_npts.at(recon_id) : 0;
    if (triangulated_features_first) {
      recon_offset[recon_id] = n_tri;
      tri_offset[recon_id]   = 0;
    } else {
      recon_offset[recon_id] = 0;
      tri_offset[recon_id]   = n_recon;
    }
  }

  // 4. Build CSR offsets for recon 3D points
  std::vector<int64_t> recon_offsets_csr(N + 1, 0);
  for (int64_t i = 0; i < N; ++i)
    recon_offsets_csr[i + 1] = recon_offsets_csr[i] + recon_track_lens[i];

  // Build CSR offsets for tri 3D points
  std::vector<int64_t> tri_offsets_csr(M + 1, 0);
  for (int64_t i = 0; i < M; ++i)
    tri_offsets_csr[i + 1] = tri_offsets_csr[i] + tri_track_lens[i];

  // Set of valid recon image IDs
  std::unordered_set<int64_t> merged_recon_ids(recon_ids.begin(), recon_ids.end());

  // 5. Remap reconstruction 3D points
  MergeResult result;
  int num_original = 0;
  for (int64_t i = 0; i < N; ++i) {
    std::vector<int64_t> new_img_ids;
    std::vector<int32_t> new_pt2d_idxs;
    bool valid = true;
    const int64_t start = recon_offsets_csr[i], end = recon_offsets_csr[i + 1];

    for (int64_t k = start; k < end; ++k) {
      const int64_t img_id = recon_track_img_ids[k];
      if (!merged_recon_ids.count(img_id)) { valid = false; break; }
      const auto off_it = recon_offset.find(img_id);
      const int offset = (off_it != recon_offset.end()) ? off_it->second : 0;
      new_img_ids.push_back(img_id);
      new_pt2d_idxs.push_back((int32_t)(recon_track_pt2d_idxs[k] + offset));
    }

    if (valid && (int64_t)new_img_ids.size() >= 2) {
      result.merged_xyz.push_back(recon_xyz[i]);
      for (size_t k = 0; k < new_img_ids.size(); ++k) {
        result.merged_track_img_ids.push_back(new_img_ids[k]);
        result.merged_track_pt2d_idxs.push_back(new_pt2d_idxs[k]);
      }
      result.merged_track_lens.push_back((int32_t)new_img_ids.size());
      num_original++;
    }
  }

  // 6. Remap triangulated 3D points
  int num_merged = 0;
  for (int64_t i = 0; i < M; ++i) {
    std::vector<int64_t> new_img_ids;
    std::vector<int32_t> new_pt2d_idxs;
    bool valid = true;
    const int64_t start = tri_offsets_csr[i], end = tri_offsets_csr[i + 1];

    for (int64_t k = start; k < end; ++k) {
      const int64_t tri_img_id = tri_track_img_ids[k];
      const auto remap_it = tri_to_recon_id.find(tri_img_id);
      if (remap_it == tri_to_recon_id.end()) { valid = false; break; }
      const int64_t recon_img_id = remap_it->second;
      if (!merged_recon_ids.count(recon_img_id)) { valid = false; break; }
      const auto off_it = tri_offset.find(recon_img_id);
      const int offset = (off_it != tri_offset.end()) ? off_it->second : 0;
      new_img_ids.push_back(recon_img_id);
      new_pt2d_idxs.push_back((int32_t)(tri_track_pt2d_idxs[k] + offset));
    }

    if (valid && (int64_t)new_img_ids.size() >= 2) {
      result.merged_xyz.push_back(tri_xyz[i]);
      for (size_t k = 0; k < new_img_ids.size(); ++k) {
        result.merged_track_img_ids.push_back(new_img_ids[k]);
        result.merged_track_pt2d_idxs.push_back(new_pt2d_idxs[k]);
      }
      result.merged_track_lens.push_back((int32_t)new_img_ids.size());
      num_merged++;
    }
  }

  std::cout << "Merged " << num_original << " original + " << num_merged
            << " triangulated 3D points, total " << result.merged_xyz.size() << std::endl;

  // 7. Update virtual_point_start
  for (auto& [recon_id, start_idx] : virtual_point_start) {
    const auto off_it = recon_offset.find(recon_id);
    if (off_it != recon_offset.end())
      start_idx += off_it->second;
  }

  // 8. Update negative_depth_observations
  for (auto& [recon_id, neg_indices] : negative_depth_observations) {
    const auto off_it = recon_offset.find(recon_id);
    if (off_it != recon_offset.end() && off_it->second != 0) {
      const int offset = off_it->second;
      std::unordered_set<int> shifted;
      shifted.reserve(neg_indices.size());
      for (const int idx : neg_indices) shifted.insert(idx + offset);
      neg_indices = std::move(shifted);
    }
  }

  result.recon_offsets               = std::move(recon_offset);
  result.tri_to_recon_id             = std::move(tri_to_recon_id);
  result.tri_offsets                 = std::move(tri_offset);
  result.virtual_point_start         = std::move(virtual_point_start);
  result.negative_depth_observations = std::move(negative_depth_observations);
  return result;
}

// ── Numpy wrapper ────────────────────────────────────────────────────────────

py::dict ComputeMergeDataWrapper(
    const std::vector<std::string>&                      tri_names,
    py::array_t<int64_t, py::array::c_style>             tri_ids,
    py::array_t<int32_t, py::array::c_style>             tri_n_pts2d,
    const std::vector<std::string>&                      recon_names,
    py::array_t<int64_t, py::array::c_style>             recon_ids,
    py::array_t<int32_t, py::array::c_style>             recon_n_pts2d,
    py::array_t<double,  py::array::c_style>             recon_xyz,
    py::array_t<int64_t, py::array::c_style>             recon_track_img_ids,
    py::array_t<int32_t, py::array::c_style>             recon_track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style>             recon_track_lens,
    py::array_t<double,  py::array::c_style>             tri_xyz,
    py::array_t<int64_t, py::array::c_style>             tri_track_img_ids,
    py::array_t<int32_t, py::array::c_style>             tri_track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style>             tri_track_lens,
    std::unordered_map<int64_t, int>                     virtual_point_start,
    std::unordered_map<int64_t, std::unordered_set<int>> negative_depth_observations,
    bool triangulated_features_first) {

  // numpy → vectors: 1D arrays via pointer-range constructor
  std::vector<int64_t> tri_ids_vec(tri_ids.data(), tri_ids.data() + tri_ids.size());
  std::vector<int32_t> tri_n_pts2d_vec(tri_n_pts2d.data(), tri_n_pts2d.data() + tri_n_pts2d.size());
  std::vector<int64_t> recon_ids_vec(recon_ids.data(), recon_ids.data() + recon_ids.size());
  std::vector<int32_t> recon_n_pts2d_vec(recon_n_pts2d.data(), recon_n_pts2d.data() + recon_n_pts2d.size());
  std::vector<int64_t> recon_img_ids_vec(recon_track_img_ids.data(), recon_track_img_ids.data() + recon_track_img_ids.size());
  std::vector<int32_t> recon_pt2d_vec(recon_track_pt2d_idxs.data(), recon_track_pt2d_idxs.data() + recon_track_pt2d_idxs.size());
  std::vector<int32_t> recon_lens_vec(recon_track_lens.data(), recon_track_lens.data() + recon_track_lens.size());
  std::vector<int64_t> tri_img_ids_vec(tri_track_img_ids.data(), tri_track_img_ids.data() + tri_track_img_ids.size());
  std::vector<int32_t> tri_pt2d_vec(tri_track_pt2d_idxs.data(), tri_track_pt2d_idxs.data() + tri_track_pt2d_idxs.size());
  std::vector<int32_t> tri_lens_vec(tri_track_lens.data(), tri_track_lens.data() + tri_track_lens.size());

  // 2D xyz arrays → vector<array<double,3>>
  auto recon_xyz_r = recon_xyz.unchecked<2>();
  std::vector<std::array<double,3>> recon_xyz_vec(recon_xyz.shape(0));
  for (int64_t i = 0; i < recon_xyz.shape(0); ++i)
    recon_xyz_vec[i] = {recon_xyz_r(i,0), recon_xyz_r(i,1), recon_xyz_r(i,2)};

  auto tri_xyz_r = tri_xyz.unchecked<2>();
  std::vector<std::array<double,3>> tri_xyz_vec(tri_xyz.shape(0));
  for (int64_t i = 0; i < tri_xyz.shape(0); ++i)
    tri_xyz_vec[i] = {tri_xyz_r(i,0), tri_xyz_r(i,1), tri_xyz_r(i,2)};

  // call core
  MergeResult r = MergeReconstructionData(
      tri_names, tri_ids_vec, tri_n_pts2d_vec,
      recon_names, recon_ids_vec, recon_n_pts2d_vec,
      recon_xyz_vec, recon_img_ids_vec, recon_pt2d_vec, recon_lens_vec,
      tri_xyz_vec, tri_img_ids_vec, tri_pt2d_vec, tri_lens_vec,
      virtual_point_start, negative_depth_observations,
      triangulated_features_first);

  // pack 2D xyz output
  const int64_t K = (int64_t)r.merged_xyz.size();
  py::array_t<double> out_xyz({K, (int64_t)3});
  auto out_xyz_w = out_xyz.mutable_unchecked<2>();
  for (int64_t i = 0; i < K; ++i) {
    out_xyz_w(i,0) = r.merged_xyz[i][0];
    out_xyz_w(i,1) = r.merged_xyz[i][1];
    out_xyz_w(i,2) = r.merged_xyz[i][2];
  }

  py::dict result;
  result["merged_xyz"]                  = out_xyz;
  result["merged_track_img_ids"]        = VecToArray1D(std::move(r.merged_track_img_ids));
  result["merged_track_pt2d_idxs"]      = VecToArray1D(std::move(r.merged_track_pt2d_idxs));
  result["merged_track_lens"]           = VecToArray1D(std::move(r.merged_track_lens));
  result["recon_offsets"]               = r.recon_offsets;
  result["tri_to_recon_id"]             = r.tri_to_recon_id;
  result["tri_offsets"]                 = r.tri_offsets;
  result["virtual_point_start"]         = r.virtual_point_start;
  result["negative_depth_observations"] = r.negative_depth_observations;
  return result;
}
