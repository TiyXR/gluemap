#include "pybind_utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int64_t FindRoot(const std::atomic<int64_t> *parents, int64_t node) {
  int64_t parent = parents[node].load(std::memory_order_relaxed);
  while (parent != node) {
    node = parent;
    parent = parents[node].load(std::memory_order_relaxed);
  }
  return node;
}

bool AtomicMin(std::atomic<int64_t> &target, int64_t value) {
  int64_t current = target.load(std::memory_order_relaxed);
  while (value < current) {
    if (target.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                     std::memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

std::vector<int64_t>
ComputeConnectedComponents(int64_t node_count, const int64_t *edge_first,
                           const int64_t *edge_second, int64_t edge_count) {
  if (node_count < 0 || edge_count < 0) {
    throw std::invalid_argument("component graph size is invalid");
  }
  auto parents = std::make_unique<std::atomic<int64_t>[]>(node_count);
#pragma omp parallel for schedule(static)
  for (int64_t node = 0; node < node_count; ++node) {
    parents[node].store(node, std::memory_order_relaxed);
  }

  std::atomic<bool> changed{true};
  while (changed.exchange(false, std::memory_order_relaxed)) {
#pragma omp parallel for schedule(static)
    for (int64_t edge = 0; edge < edge_count; ++edge) {
      const int64_t first = edge_first[edge];
      const int64_t second = edge_second[edge];
      if (first < 0 || first >= node_count || second < 0 ||
          second >= node_count) {
        continue;
      }
      const int64_t first_root = FindRoot(parents.get(), first);
      const int64_t second_root = FindRoot(parents.get(), second);
      if (first_root == second_root) {
        continue;
      }
      const int64_t high = std::max(first_root, second_root);
      const int64_t low = std::min(first_root, second_root);
      if (AtomicMin(parents[high], low)) {
        changed.store(true, std::memory_order_relaxed);
      }
    }
#pragma omp parallel for schedule(static)
    for (int64_t node = 0; node < node_count; ++node) {
      const int64_t root = FindRoot(parents.get(), node);
      if (parents[node].exchange(root, std::memory_order_relaxed) != root) {
        changed.store(true, std::memory_order_relaxed);
      }
    }
  }

  std::vector<int64_t> labels(node_count);
#pragma omp parallel for schedule(static)
  for (int64_t node = 0; node < node_count; ++node) {
    labels[node] = FindRoot(parents.get(), node);
  }
  return labels;
}

int64_t SpatialBucketKey(int64_t column, int64_t row) {
  const uint64_t high = static_cast<uint64_t>(static_cast<uint32_t>(column));
  const uint64_t low = static_cast<uint64_t>(static_cast<uint32_t>(row));
  return static_cast<int64_t>((high << 32) | low);
}

std::vector<int64_t> BatchSpatialIntern(
    const int64_t *existing_frames, const double *existing_x,
    const double *existing_y, const std::vector<std::string> &existing_uids,
    int64_t existing_count, const int64_t *incoming_frames,
    const double *incoming_x, const double *incoming_y,
    const std::vector<std::string> &incoming_uids, int64_t incoming_count,
    double radius) {
  std::map<int64_t, std::vector<int64_t>> existing_by_frame;
  std::map<int64_t, std::vector<int64_t>> incoming_by_frame;
  for (int64_t index = 0; index < existing_count; ++index) {
    existing_by_frame[existing_frames[index]].push_back(index);
  }
  for (int64_t index = 0; index < incoming_count; ++index) {
    incoming_by_frame[incoming_frames[index]].push_back(index);
  }
  std::vector<std::pair<int64_t, std::vector<int64_t>>> frame_groups(
      incoming_by_frame.begin(), incoming_by_frame.end());
  std::vector<int64_t> representatives(incoming_count, -1);
  const double radius_squared = radius * radius;

#pragma omp parallel for schedule(dynamic)
  for (int64_t group_index = 0;
       group_index < static_cast<int64_t>(frame_groups.size()); ++group_index) {
    const int64_t frame = frame_groups[group_index].first;
    const auto &incoming_indexes = frame_groups[group_index].second;
    std::unordered_map<int64_t, std::vector<int64_t>> buckets;
    const auto existing_it = existing_by_frame.find(frame);
    if (existing_it != existing_by_frame.end()) {
      for (const int64_t index : existing_it->second) {
        const int64_t column = static_cast<int64_t>(std::floor(existing_x[index] / radius));
        const int64_t row = static_cast<int64_t>(std::floor(existing_y[index] / radius));
        buckets[SpatialBucketKey(column, row)].push_back(index);
      }
    }
    for (const int64_t incoming_index : incoming_indexes) {
      const int64_t column = static_cast<int64_t>(
          std::floor(incoming_x[incoming_index] / radius));
      const int64_t row = static_cast<int64_t>(
          std::floor(incoming_y[incoming_index] / radius));
      double best_distance = std::numeric_limits<double>::infinity();
      int64_t best = -1;
      const std::string *best_uid = nullptr;
      for (int64_t column_delta = -1; column_delta <= 1; ++column_delta) {
        for (int64_t row_delta = -1; row_delta <= 1; ++row_delta) {
          const auto bucket_it = buckets.find(SpatialBucketKey(
              column + column_delta, row + row_delta));
          if (bucket_it == buckets.end()) {
            continue;
          }
          for (const int64_t encoded : bucket_it->second) {
            const bool is_existing = encoded < existing_count;
            const int64_t source_index =
                is_existing ? encoded : encoded - existing_count;
            const double candidate_x =
                is_existing ? existing_x[source_index] : incoming_x[source_index];
            const double candidate_y =
                is_existing ? existing_y[source_index] : incoming_y[source_index];
            const std::string &candidate_uid = is_existing
                                                   ? existing_uids[source_index]
                                                   : incoming_uids[source_index];
            const double dx = candidate_x - incoming_x[incoming_index];
            const double dy = candidate_y - incoming_y[incoming_index];
            const double distance = dx * dx + dy * dy;
            if (distance <= radius_squared &&
                (distance < best_distance ||
                 (distance == best_distance &&
                  (best_uid == nullptr || candidate_uid < *best_uid)))) {
              best_distance = distance;
              best = encoded;
              best_uid = &candidate_uid;
            }
          }
        }
      }
      if (best < 0) {
        best = existing_count + incoming_index;
        buckets[SpatialBucketKey(column, row)].push_back(best);
      }
      representatives[incoming_index] = best;
    }
  }
  return representatives;
}

} // namespace

py::array_t<int64_t> ComputeConnectedComponentsWrapper(
    int64_t node_count,
    py::array_t<int64_t, py::array::c_style> edge_first,
    py::array_t<int64_t, py::array::c_style> edge_second) {
  if (edge_first.size() != edge_second.size()) {
    throw std::invalid_argument("component edge arrays differ in length");
  }
  std::vector<int64_t> labels;
  {
    py::gil_scoped_release release;
    labels = ComputeConnectedComponents(node_count, edge_first.data(),
                                        edge_second.data(), edge_first.size());
  }
  return VecToArray1D(std::move(labels));
}

py::array_t<int64_t> BatchSpatialInternWrapper(
    py::array_t<int64_t, py::array::c_style> existing_frames,
    py::array_t<double, py::array::c_style> existing_x,
    py::array_t<double, py::array::c_style> existing_y,
    const std::vector<std::string> &existing_uids,
    py::array_t<int64_t, py::array::c_style> incoming_frames,
    py::array_t<double, py::array::c_style> incoming_x,
    py::array_t<double, py::array::c_style> incoming_y,
    const std::vector<std::string> &incoming_uids, double radius) {
  if (existing_frames.size() != existing_x.size() ||
      existing_frames.size() != existing_y.size() ||
      existing_frames.size() != static_cast<int64_t>(existing_uids.size()) ||
      incoming_frames.size() != incoming_x.size() ||
      incoming_frames.size() != incoming_y.size() ||
      incoming_frames.size() != static_cast<int64_t>(incoming_uids.size())) {
    throw std::invalid_argument("spatial intern arrays differ in length");
  }
  std::vector<int64_t> representatives;
  {
    py::gil_scoped_release release;
    representatives = BatchSpatialIntern(
        existing_frames.data(), existing_x.data(), existing_y.data(),
        existing_uids, existing_frames.size(), incoming_frames.data(),
        incoming_x.data(), incoming_y.data(), incoming_uids,
        incoming_frames.size(), radius);
  }
  return VecToArray1D(std::move(representatives));
}
