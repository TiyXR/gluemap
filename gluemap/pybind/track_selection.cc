#include "pybind_utils.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

static inline uint64_t CanonicalPairKey(int64_t id1, int64_t id2) {
  const uint64_t lo = std::min((uint64_t)id1, (uint64_t)id2);
  const uint64_t hi = std::max((uint64_t)id1, (uint64_t)id2);
  return (lo << 32) | hi;
}

//
// Args (CSR-like format):
//   point3d_ids:      (N,)           int64  – all point3D IDs
//   track_image_ids:  (total_elems,) int64  – flattened track image IDs
//   track_pt2d_idxs:  (total_elems,) int64  – flattened track point2D indices
//   track_lengths:    (N,)           int32  – number of elements per track
//   virtual_point_start: {image_id -> int}
//   sift_count:          {image_id -> int}
//
// Returns: vector of point3D IDs that should be deleted.
std::vector<int64_t> SelectTracksToDelete(
    const std::vector<int64_t>& point3d_ids,
    const std::vector<int64_t>& track_image_ids,
    const std::vector<int64_t>& track_pt2d_idxs,
    const std::vector<int32_t>& track_lengths,
    const std::unordered_map<int64_t, int>& virtual_point_start,
    const std::unordered_map<int64_t, int>& sift_count,
    int min_num_support_abs,
    const std::string& tracks_to_keep) {

  const int64_t N = (int64_t)point3d_ids.size();

  // Build cumulative offset array (CSR format)
  std::vector<int64_t> offsets(N + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    offsets[i + 1] = offsets[i] + track_lengths[i];
  }

  // ── Classify tracks as SIFT / prior / virtual ──
  std::vector<int64_t> sift_idxs, prior_idxs, virtual_idxs;

  for (int64_t i = 0; i < N; ++i) {
    bool is_sift = true;
    bool is_virtual = false;
    for (int64_t k = offsets[i]; k < offsets[i + 1]; ++k) {
      const int64_t img_id = track_image_ids[k];
      const int64_t pt2d_idx = track_pt2d_idxs[k];

      const auto sc_it = sift_count.find(img_id);
      const int s_count = (sc_it != sift_count.end()) ? sc_it->second : 0;
      const auto vp_it = virtual_point_start.find(img_id);
      const int vp_start =
          (vp_it != virtual_point_start.end()) ? vp_it->second : INT32_MAX;

      if (static_cast<int>(pt2d_idx) >= s_count) is_sift = false;
      if (static_cast<int>(pt2d_idx) >= vp_start) is_virtual = true;
    }

    if (is_sift) sift_idxs.push_back(i);
    else if (is_virtual) virtual_idxs.push_back(i);
    else prior_idxs.push_back(i);
  }

  std::cout << "Track classification: " << sift_idxs.size() << " SIFT, "
            << prior_idxs.size() << " prior, " << virtual_idxs.size()
            << " virtual" << std::endl;

  // ── Count pair coverage from SIFT tracks ──
  std::unordered_map<uint64_t, int> pair_count;
  for (const int64_t idx : sift_idxs) {
    const int64_t start = offsets[idx], end = offsets[idx + 1];
    for (int64_t i = start; i < end; ++i) {
      for (int64_t j = i + 1; j < end; ++j) {
        pair_count[CanonicalPairKey(track_image_ids[i], track_image_ids[j])] += 1;
      }
    }
  }

  // ── Shuffle prior and virtual index lists ──
  std::mt19937 rng(42);
  std::shuffle(prior_idxs.begin(), prior_idxs.end(), rng);
  std::shuffle(virtual_idxs.begin(), virtual_idxs.end(), rng);

  const bool do_prior = tracks_to_keep.find("prior") != std::string::npos;
  const bool do_virtual = tracks_to_keep.find("virtual") != std::string::npos;

  std::vector<int64_t> ids_to_remove;
  size_t num_prior_selected = 0, num_virtual_selected = 0;

  struct Pass {
    bool is_virtual;
    std::vector<int64_t>& idx_list;
    bool do_pass;
  };
  Pass passes[] = {{false, prior_idxs, do_prior}, {true, virtual_idxs, do_virtual}};

  for (auto& pass : passes) {
    if (pass.do_pass) {
      if (pass.is_virtual) num_virtual_selected += pass.idx_list.size();
      else num_prior_selected += pass.idx_list.size();
      continue;
    }

    for (const int64_t idx : pass.idx_list) {
      const int64_t start = offsets[idx], end = offsets[idx + 1];

      bool hit = false;
      for (int64_t i = start; i < end && !hit; ++i) {
        for (int64_t j = i + 1; j < end; ++j) {
          const auto key = CanonicalPairKey(track_image_ids[i], track_image_ids[j]);
          const auto it = pair_count.find(key);
          if ((it != pair_count.end() ? it->second : 0) <= min_num_support_abs) {
            hit = true;
            break;
          }
        }
      }

      if (hit) {
        if (pass.is_virtual) num_virtual_selected++;
        else num_prior_selected++;
        for (int64_t i = start; i < end; ++i) {
          for (int64_t j = i + 1; j < end; ++j) {
            pair_count[CanonicalPairKey(track_image_ids[i], track_image_ids[j])] += 1;
          }
        }
      } else {
        ids_to_remove.push_back(point3d_ids[idx]);
      }
    }
  }

  std::cout << "SelectTrack: kept " << sift_idxs.size() << " SIFT + "
            << num_prior_selected << "/" << prior_idxs.size() << " prior + "
            << num_virtual_selected << "/" << virtual_idxs.size()
            << " virtual, removed " << ids_to_remove.size() << std::endl;

  return ids_to_remove;
}

// ── Numpy wrapper ────────────────────────────────────────────────────────────
py::array_t<int64_t> ComputeTracksToDeleteWrapper(
    py::array_t<int64_t, py::array::c_style> point3d_ids,
    py::array_t<int64_t, py::array::c_style> track_image_ids,
    py::array_t<int64_t, py::array::c_style> track_pt2d_idxs,
    py::array_t<int32_t, py::array::c_style> track_lengths,
    const std::unordered_map<int64_t, int>& virtual_point_start,
    const std::unordered_map<int64_t, int>& sift_count,
    int min_num_support_abs,
    const std::string& tracks_to_keep) {

  // numpy → vector (pointer-range constructor, no element-wise copy)
  std::vector<int64_t> ids_vec(point3d_ids.data(), point3d_ids.data() + point3d_ids.size());
  std::vector<int64_t> img_ids_vec(track_image_ids.data(), track_image_ids.data() + track_image_ids.size());
  std::vector<int64_t> pt2d_vec(track_pt2d_idxs.data(), track_pt2d_idxs.data() + track_pt2d_idxs.size());
  std::vector<int32_t> lens_vec(track_lengths.data(), track_lengths.data() + track_lengths.size());

  // call core
  std::vector<int64_t> to_delete = SelectTracksToDelete(
      ids_vec, img_ids_vec, pt2d_vec, lens_vec,
      virtual_point_start, sift_count, min_num_support_abs, tracks_to_keep);

  return VecToArray1D(std::move(to_delete));
}
