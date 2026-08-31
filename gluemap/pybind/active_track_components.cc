#include "pybind_utils.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <openssl/sha.h>

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

struct ActiveTrackCandidate {
  int64_t component_index;
  int64_t time_bin;
  int64_t row;
  int64_t column;
};

struct ActiveTrackIntervalSelection {
  std::vector<int64_t> component_indexes;
  std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> bucket_counts;
  std::string selected_uids_sha256;
  std::string bridge_uids_sha256;
  int64_t candidate_count = 0;
  int64_t bridge_count = 0;
  int64_t insufficient_view_count = 0;
  int64_t insufficient_parallax_count = 0;
};

struct ActiveTrackSelectionBatch {
  std::vector<int64_t> selected_component_indexes;
  std::vector<int64_t> selected_offsets;
  std::vector<int64_t> candidate_counts;
  std::vector<int64_t> bridge_counts;
  std::vector<std::string> selected_uids_sha256;
  std::vector<std::string> bridge_uids_sha256;
  std::vector<int64_t> insufficient_view_counts;
  std::vector<int64_t> insufficient_parallax_counts;
  std::vector<int64_t> bucket_offsets;
  std::vector<int64_t> bucket_time_bins;
  std::vector<int64_t> bucket_rows;
  std::vector<int64_t> bucket_columns;
  std::vector<int64_t> bucket_counts;
};

std::string HexSha256(const std::string &payload);

std::string CanonicalStringListSha256(
    const std::vector<int64_t> &component_indexes,
    const std::vector<std::string> &component_uids,
    const int64_t *integer_metrics, int64_t interval_count, int64_t interval,
    bool bridge_only) {
  std::string payload;
  payload.reserve(component_indexes.size() * 68 + 2);
  payload.push_back('[');
  bool first_value = true;
  constexpr char hex[] = "0123456789abcdef";
  for (const int64_t component_index : component_indexes) {
    const int64_t integer_base =
        (component_index * interval_count + interval) * 5;
    if (bridge_only && integer_metrics[integer_base + 1] == 0) {
      continue;
    }
    if (!first_value) {
      payload.push_back(',');
    }
    first_value = false;
    payload.push_back('"');
    for (const unsigned char value :
         component_uids[static_cast<size_t>(component_index)]) {
      switch (value) {
      case '"':
        payload.append("\\\"");
        break;
      case '\\':
        payload.append("\\\\");
        break;
      case '\b':
        payload.append("\\b");
        break;
      case '\f':
        payload.append("\\f");
        break;
      case '\n':
        payload.append("\\n");
        break;
      case '\r':
        payload.append("\\r");
        break;
      case '\t':
        payload.append("\\t");
        break;
      default:
        if (value < 0x20) {
          payload.append("\\u00");
          payload.push_back(hex[value >> 4]);
          payload.push_back(hex[value & 0x0f]);
        } else {
          payload.push_back(static_cast<char>(value));
        }
      }
    }
    payload.push_back('"');
  }
  payload.push_back(']');
  return HexSha256(payload);
}

ActiveTrackSelectionBatch SelectActiveTrackCandidates(
    const int64_t *integer_metrics, const double *float_metrics,
    const std::vector<std::string> &component_uids, int64_t component_count,
    int64_t interval_count, int64_t minimum_track_views,
    double minimum_parallax_diagonals, int64_t maximum_active_tracks,
    int64_t maximum_tracks_per_grid_cell, int native_thread_count) {
  std::vector<ActiveTrackIntervalSelection> interval_results(
      static_cast<size_t>(interval_count));

#pragma omp parallel for schedule(dynamic) num_threads(native_thread_count)
  for (int64_t interval = 0; interval < interval_count; ++interval) {
    auto &result = interval_results[static_cast<size_t>(interval)];
    std::vector<ActiveTrackCandidate> candidates;
    candidates.reserve(static_cast<size_t>(component_count));
    for (int64_t component = 0; component < component_count; ++component) {
      const int64_t integer_base = (component * interval_count + interval) * 5;
      const int64_t float_base = (component * interval_count + interval) * 2;
      const int64_t view_count = integer_metrics[integer_base];
      const double parallax = float_metrics[float_base];
      const bool insufficient_views = view_count < minimum_track_views;
      const bool insufficient_parallax =
          parallax < minimum_parallax_diagonals;
      result.insufficient_view_count += insufficient_views ? 1 : 0;
      result.insufficient_parallax_count += insufficient_parallax ? 1 : 0;
      if (insufficient_views || insufficient_parallax) {
        continue;
      }
      candidates.push_back({component, integer_metrics[integer_base + 2],
                            integer_metrics[integer_base + 3],
                            integer_metrics[integer_base + 4]});
    }
    result.candidate_count = static_cast<int64_t>(candidates.size());

    const auto quality_less = [&](const ActiveTrackCandidate &first,
                                  const ActiveTrackCandidate &second) {
      const int64_t first_integer =
          (first.component_index * interval_count + interval) * 5;
      const int64_t second_integer =
          (second.component_index * interval_count + interval) * 5;
      const int64_t first_float =
          (first.component_index * interval_count + interval) * 2;
      const int64_t second_float =
          (second.component_index * interval_count + interval) * 2;
      const auto first_bucket =
          std::tie(first.time_bin, first.row, first.column);
      const auto second_bucket =
          std::tie(second.time_bin, second.row, second.column);
      if (first_bucket != second_bucket) {
        return first_bucket < second_bucket;
      }
      const bool first_bridge = integer_metrics[first_integer + 1] != 0;
      const bool second_bridge = integer_metrics[second_integer + 1] != 0;
      if (first_bridge != second_bridge) {
        return first_bridge;
      }
      const int64_t first_views = integer_metrics[first_integer];
      const int64_t second_views = integer_metrics[second_integer];
      if (first_views != second_views) {
        return first_views > second_views;
      }
      const double first_parallax = float_metrics[first_float];
      const double second_parallax = float_metrics[second_float];
      if (first_parallax != second_parallax) {
        return first_parallax > second_parallax;
      }
      const double first_score = float_metrics[first_float + 1];
      const double second_score = float_metrics[second_float + 1];
      if (first_score != second_score) {
        return first_score > second_score;
      }
      return component_uids[static_cast<size_t>(first.component_index)] <
             component_uids[static_cast<size_t>(second.component_index)];
    };
    std::sort(candidates.begin(), candidates.end(), quality_less);

    std::vector<std::pair<size_t, size_t>> bucket_ranges;
    for (size_t begin = 0; begin < candidates.size();) {
      size_t end = begin + 1;
      while (end < candidates.size() &&
             candidates[end].time_bin == candidates[begin].time_bin &&
             candidates[end].row == candidates[begin].row &&
             candidates[end].column == candidates[begin].column) {
        ++end;
      }
      bucket_ranges.emplace_back(begin, end);
      begin = end;
    }

    result.component_indexes.reserve(static_cast<size_t>(std::min(
        maximum_active_tracks, static_cast<int64_t>(candidates.size()))));
    for (int64_t depth = 0;
         depth < maximum_tracks_per_grid_cell &&
         static_cast<int64_t>(result.component_indexes.size()) <
             maximum_active_tracks;
         ++depth) {
      bool progressed = false;
      for (const auto &[begin, end] : bucket_ranges) {
        const size_t candidate_index = begin + static_cast<size_t>(depth);
        if (candidate_index >= end) {
          continue;
        }
        result.component_indexes.push_back(
            candidates[candidate_index].component_index);
        progressed = true;
        if (static_cast<int64_t>(result.component_indexes.size()) >=
            maximum_active_tracks) {
          break;
        }
      }
      if (!progressed) {
        break;
      }
    }
    for (const int64_t component_index : result.component_indexes) {
      const int64_t integer_base =
          (component_index * interval_count + interval) * 5;
      result.bridge_count += integer_metrics[integer_base + 1] != 0 ? 1 : 0;
      ++result.bucket_counts[std::make_tuple(integer_metrics[integer_base + 2],
                                             integer_metrics[integer_base + 3],
                                             integer_metrics[integer_base + 4])];
    }
    result.selected_uids_sha256 = CanonicalStringListSha256(
        result.component_indexes, component_uids, integer_metrics,
        interval_count, interval, false);
    result.bridge_uids_sha256 = CanonicalStringListSha256(
        result.component_indexes, component_uids, integer_metrics,
        interval_count, interval, true);
  }

  ActiveTrackSelectionBatch batch;
  batch.selected_offsets.reserve(static_cast<size_t>(interval_count + 1));
  batch.candidate_counts.reserve(static_cast<size_t>(interval_count));
  batch.bridge_counts.reserve(static_cast<size_t>(interval_count));
  batch.selected_uids_sha256.reserve(static_cast<size_t>(interval_count));
  batch.bridge_uids_sha256.reserve(static_cast<size_t>(interval_count));
  batch.insufficient_view_counts.reserve(static_cast<size_t>(interval_count));
  batch.insufficient_parallax_counts.reserve(
      static_cast<size_t>(interval_count));
  batch.selected_offsets.push_back(0);
  batch.bucket_offsets.push_back(0);
  for (auto &result : interval_results) {
    batch.selected_component_indexes.insert(
        batch.selected_component_indexes.end(), result.component_indexes.begin(),
        result.component_indexes.end());
    batch.selected_offsets.push_back(
        static_cast<int64_t>(batch.selected_component_indexes.size()));
    batch.candidate_counts.push_back(result.candidate_count);
    batch.bridge_counts.push_back(result.bridge_count);
    batch.selected_uids_sha256.push_back(
        std::move(result.selected_uids_sha256));
    batch.bridge_uids_sha256.push_back(
        std::move(result.bridge_uids_sha256));
    batch.insufficient_view_counts.push_back(result.insufficient_view_count);
    batch.insufficient_parallax_counts.push_back(
        result.insufficient_parallax_count);
    for (const auto &[bucket, count] : result.bucket_counts) {
      batch.bucket_time_bins.push_back(std::get<0>(bucket));
      batch.bucket_rows.push_back(std::get<1>(bucket));
      batch.bucket_columns.push_back(std::get<2>(bucket));
      batch.bucket_counts.push_back(count);
    }
    batch.bucket_offsets.push_back(
        static_cast<int64_t>(batch.bucket_counts.size()));
  }
  return batch;
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
  std::vector<int64_t> representatives(incoming_count);
#pragma omp parallel for schedule(static)
  for (int64_t index = 0; index < incoming_count; ++index) {
    representatives[index] = existing_count + index;
  }
  const double radius_squared = radius * radius;

  struct Candidate {
    double distance;
    int64_t existing_index;
    int64_t incoming_index;
  };

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
    std::vector<Candidate> candidates;
    for (const int64_t incoming_index : incoming_indexes) {
      const int64_t column = static_cast<int64_t>(
          std::floor(incoming_x[incoming_index] / radius));
      const int64_t row = static_cast<int64_t>(
          std::floor(incoming_y[incoming_index] / radius));
      for (int64_t column_delta = -1; column_delta <= 1; ++column_delta) {
        for (int64_t row_delta = -1; row_delta <= 1; ++row_delta) {
          const auto bucket_it = buckets.find(SpatialBucketKey(
              column + column_delta, row + row_delta));
          if (bucket_it == buckets.end()) {
            continue;
          }
          for (const int64_t existing_index : bucket_it->second) {
            const double dx =
                existing_x[existing_index] - incoming_x[incoming_index];
            const double dy =
                existing_y[existing_index] - incoming_y[incoming_index];
            const double distance = dx * dx + dy * dy;
            if (distance <= radius_squared) {
              candidates.push_back(
                  {distance, existing_index, incoming_index});
            }
          }
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [&](const Candidate &left, const Candidate &right) {
                if (left.distance != right.distance) {
                  return left.distance < right.distance;
                }
                const auto &left_existing =
                    existing_uids[left.existing_index];
                const auto &right_existing =
                    existing_uids[right.existing_index];
                if (left_existing != right_existing) {
                  return left_existing < right_existing;
                }
                return incoming_uids[left.incoming_index] <
                       incoming_uids[right.incoming_index];
              });
    std::unordered_map<int64_t, bool> used_existing;
    std::unordered_map<int64_t, bool> assigned_incoming;
    for (const Candidate &candidate : candidates) {
      if (used_existing.find(candidate.existing_index) != used_existing.end() ||
          assigned_incoming.find(candidate.incoming_index) !=
              assigned_incoming.end()) {
        continue;
      }
      representatives[candidate.incoming_index] = candidate.existing_index;
      used_existing.emplace(candidate.existing_index, true);
      assigned_incoming.emplace(candidate.incoming_index, true);
    }
  }
  return representatives;
}

std::string HexSha256(const std::string &payload) {
  unsigned char digest[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<const unsigned char *>(payload.data()),
         payload.size(), digest);
  constexpr char hex[] = "0123456789abcdef";
  std::string result(SHA256_DIGEST_LENGTH * 2, '0');
  for (size_t index = 0; index < SHA256_DIGEST_LENGTH; ++index) {
    result[index * 2] = hex[digest[index] >> 4];
    result[index * 2 + 1] = hex[digest[index] & 0x0f];
  }
  return result;
}

std::vector<std::string> BatchObservationUids(
    const std::string &prediction_uid, const int64_t *track_indexes,
    const int64_t *view_indexes, const std::vector<std::string> &frame_uids,
    int64_t count) {
  std::vector<std::string> result(count);
#pragma omp parallel for schedule(static)
  for (int64_t index = 0; index < count; ++index) {
    std::string payload = "jarailsense.gluemap-track-observation/v1";
    payload.push_back('\0');
    payload.append(prediction_uid);
    payload.push_back('\0');
    payload.append(std::to_string(track_indexes[index]));
    payload.push_back('\0');
    payload.append(std::to_string(view_indexes[index]));
    payload.push_back('\0');
    payload.append(frame_uids[index]);
    result[index] = HexSha256(payload);
  }
  return result;
}

class ActiveTrackGraph {
public:
  int64_t AddNodes(const std::vector<std::string> &uids) {
    py::gil_scoped_release release;
    AddNodesImpl(uids, nullptr);
    return static_cast<int64_t>(uids.size());
  }

  py::array_t<int64_t>
  AddNodesWithRows(const std::vector<std::string> &uids) {
    std::vector<int64_t> rows;
    {
      py::gil_scoped_release release;
      rows.reserve(uids.size());
      AddNodesImpl(uids, &rows);
    }
    return VecToArray1D(std::move(rows));
  }

  int64_t AddNodesImpl(const std::vector<std::string> &uids,
                       std::vector<int64_t> *rows) {
    for (const std::string &uid : uids) {
      if (uid.empty() || node_by_uid_.find(uid) != node_by_uid_.end()) {
        throw std::invalid_argument("active track graph node is invalid");
      }
      if (uid_by_node_.size() >=
          static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("active track graph node limit exceeded");
      }
      const int64_t node = static_cast<int64_t>(uid_by_node_.size());
      node_by_uid_.emplace(uid, node);
      uid_by_node_.push_back(uid);
      parents_.push_back(node);
      component_uids_.push_back(uid);
      alive_.push_back(true);
      int64_t tensor_row = -1;
      if (!free_tensor_rows_.empty()) {
        tensor_row = free_tensor_rows_.back();
        free_tensor_rows_.pop_back();
      } else {
        tensor_row = next_tensor_row_++;
      }
      tensor_rows_.push_back(tensor_row);
      if (rows != nullptr) {
        rows->push_back(tensor_row);
      }
      ++live_node_count_;
    }
    return static_cast<int64_t>(uids.size());
  }

  int64_t AddEdges(const std::vector<std::string> &first_uids,
                   const std::vector<std::string> &second_uids) {
    if (first_uids.size() != second_uids.size()) {
      throw std::invalid_argument("active track graph edge columns differ");
    }
    py::gil_scoped_release release;
    int64_t inserted = 0;
    for (size_t index = 0; index < first_uids.size(); ++index) {
      inserted += AddEdge(Node(first_uids[index]), Node(second_uids[index]));
    }
    return inserted;
  }

  int64_t AddStarEdges(
      py::array_t<int64_t, py::array::c_style> track_indexes,
      py::array_t<int64_t, py::array::c_style> view_indexes,
      const std::vector<std::string> &resolved_uids) {
    if (track_indexes.size() != view_indexes.size() ||
        track_indexes.size() != static_cast<int64_t>(resolved_uids.size())) {
      throw std::invalid_argument("active track Star columns differ");
    }
    py::gil_scoped_release release;
    const int64_t *tracks = track_indexes.data();
    const int64_t *views = view_indexes.data();
    int64_t current_track = -1;
    int64_t center_node = -1;
    int64_t inserted = 0;
    for (int64_t index = 0; index < track_indexes.size(); ++index) {
      const int64_t node = Node(resolved_uids[static_cast<size_t>(index)]);
      if (tracks[index] != current_track) {
        if (views[index] != 0) {
          throw std::invalid_argument(
              "active track Star does not start at its center");
        }
        current_track = tracks[index];
        center_node = node;
      } else {
        if (views[index] == 0 || center_node < 0) {
          throw std::invalid_argument("active track Star view is invalid");
        }
        inserted += AddEdge(center_node, node);
      }
    }
    return inserted;
  }

  std::vector<std::string>
  ComponentUids(const std::vector<std::string> &uids) {
    py::gil_scoped_release release;
    std::vector<std::string> result;
    result.reserve(uids.size());
    for (const std::string &uid : uids) {
      const int64_t node = Node(uid);
      result.push_back(component_uids_[static_cast<size_t>(Find(node))]);
    }
    return result;
  }

  py::tuple GroupComponents(const std::vector<std::string> &uids) {
    std::vector<int64_t> ordered_indexes;
    std::vector<int64_t> offsets;
    std::vector<std::string> component_uids;
    {
      py::gil_scoped_release release;
      std::unordered_map<std::string, int64_t> group_by_component;
      std::vector<std::vector<int64_t>> groups;
      groups.reserve(uids.size());
      for (size_t index = 0; index < uids.size(); ++index) {
        const int64_t root = Find(Node(uids[index]));
        const std::string &component_uid =
            component_uids_[static_cast<size_t>(root)];
        const auto inserted = group_by_component.emplace(
            component_uid, static_cast<int64_t>(groups.size()));
        if (inserted.second) {
          groups.emplace_back();
          component_uids.push_back(component_uid);
        }
        groups[static_cast<size_t>(inserted.first->second)].push_back(
            static_cast<int64_t>(index));
      }
      ordered_indexes.reserve(uids.size());
      offsets.reserve(groups.size() + 1);
      offsets.push_back(0);
      for (const auto &group : groups) {
        ordered_indexes.insert(ordered_indexes.end(), group.begin(), group.end());
        offsets.push_back(static_cast<int64_t>(ordered_indexes.size()));
      }
    }
    return py::make_tuple(VecToArray1D(std::move(ordered_indexes)),
                          VecToArray1D(std::move(offsets)),
                          std::move(component_uids));
  }

  py::tuple GroupComponentRows(
      const std::vector<std::string> &uids,
      py::array_t<int64_t, py::array::c_style> frame_ids) {
    if (frame_ids.size() != static_cast<int64_t>(uids.size())) {
      throw std::invalid_argument("active track component frame column differs");
    }
    std::vector<int64_t> ordered_indexes;
    std::vector<int64_t> ordered_rows;
    std::vector<int64_t> offsets;
    std::vector<std::string> component_uids;
    {
      py::gil_scoped_release release;
      std::unordered_map<std::string, int64_t> group_by_component;
      std::vector<std::vector<int64_t>> groups;
      std::vector<std::vector<int64_t>> group_rows;
      std::vector<std::unordered_set<int64_t>> seen_frames;
      groups.reserve(uids.size());
      group_rows.reserve(uids.size());
      seen_frames.reserve(uids.size());
      const int64_t *frames = frame_ids.data();
      for (size_t index = 0; index < uids.size(); ++index) {
        const int64_t node = Node(uids[index]);
        const int64_t root = Find(node);
        const std::string &component_uid =
            component_uids_[static_cast<size_t>(root)];
        const auto inserted = group_by_component.emplace(
            component_uid, static_cast<int64_t>(groups.size()));
        if (inserted.second) {
          groups.emplace_back();
          group_rows.emplace_back();
          seen_frames.emplace_back();
          component_uids.push_back(component_uid);
        }
        const size_t group = static_cast<size_t>(inserted.first->second);
        if (!seen_frames[group].insert(frames[index]).second) {
          continue;
        }
        groups[group].push_back(static_cast<int64_t>(index));
        group_rows[group].push_back(tensor_rows_[static_cast<size_t>(node)]);
      }
      ordered_indexes.reserve(uids.size());
      ordered_rows.reserve(uids.size());
      offsets.reserve(groups.size() + 1);
      offsets.push_back(0);
      for (size_t group = 0; group < groups.size(); ++group) {
        ordered_indexes.insert(ordered_indexes.end(), groups[group].begin(),
                               groups[group].end());
        ordered_rows.insert(ordered_rows.end(), group_rows[group].begin(),
                            group_rows[group].end());
        offsets.push_back(static_cast<int64_t>(ordered_indexes.size()));
      }
    }
    return py::make_tuple(VecToArray1D(std::move(ordered_indexes)),
                          VecToArray1D(std::move(offsets)),
                          std::move(component_uids),
                          VecToArray1D(std::move(ordered_rows)));
  }

  void RemoveNodes(const std::vector<std::string> &uids) {
    py::gil_scoped_release release;
    RemoveNodesImpl(uids, nullptr);
  }

  py::array_t<int64_t>
  RemoveNodesWithRows(const std::vector<std::string> &uids) {
    std::vector<int64_t> rows;
    {
      py::gil_scoped_release release;
      rows.reserve(uids.size());
      RemoveNodesImpl(uids, &rows);
    }
    return VecToArray1D(std::move(rows));
  }

  void RemoveNodesImpl(const std::vector<std::string> &uids,
                       std::vector<int64_t> *rows) {
    std::vector<uint8_t> remove(uid_by_node_.size(), 0);
    for (const std::string &uid : uids) {
      const int64_t node = Node(uid);
      if (remove[static_cast<size_t>(node)] != 0) {
        throw std::invalid_argument("active track graph node repeats");
      }
      remove[static_cast<size_t>(node)] = 1;
    }

    std::vector<std::string> previous_components(uid_by_node_.size());
    for (size_t node = 0; node < uid_by_node_.size(); ++node) {
      if (alive_[node]) {
        previous_components[node] = component_uids_[
            static_cast<size_t>(Find(static_cast<int64_t>(node)))];
      }
    }
    for (size_t node = 0; node < uid_by_node_.size(); ++node) {
      if (remove[node] != 0) {
        const int64_t tensor_row = tensor_rows_[node];
        free_tensor_rows_.push_back(tensor_row);
        if (rows != nullptr) {
          rows->push_back(tensor_row);
        }
        alive_[node] = false;
        node_by_uid_.erase(uid_by_node_[node]);
        --live_node_count_;
      }
      if (alive_[node]) {
        parents_[node] = static_cast<int64_t>(node);
        component_uids_[node] = previous_components[node];
      }
    }

    std::unordered_set<uint64_t> retained_edges;
    retained_edges.reserve(edges_.size());
    for (const uint64_t edge : edges_) {
      const int64_t first = static_cast<int64_t>(edge >> 32);
      const int64_t second = static_cast<int64_t>(edge & 0xffffffffULL);
      if (alive_[static_cast<size_t>(first)] &&
          alive_[static_cast<size_t>(second)]) {
        retained_edges.insert(edge);
      }
    }
    edges_ = std::move(retained_edges);
    for (const uint64_t edge : edges_) {
      Union(static_cast<int64_t>(edge >> 32),
            static_cast<int64_t>(edge & 0xffffffffULL));
    }
  }

  int64_t NodeCount() const { return live_node_count_; }
  int64_t EdgeCount() const { return static_cast<int64_t>(edges_.size()); }

private:
  int64_t Node(const std::string &uid) const {
    const auto found = node_by_uid_.find(uid);
    if (found == node_by_uid_.end() ||
        !alive_[static_cast<size_t>(found->second)]) {
      throw std::invalid_argument("active track graph node is absent");
    }
    return found->second;
  }

  int64_t Find(int64_t node) {
    int64_t root = node;
    while (parents_[static_cast<size_t>(root)] != root) {
      root = parents_[static_cast<size_t>(root)];
    }
    while (parents_[static_cast<size_t>(node)] != node) {
      const int64_t parent = parents_[static_cast<size_t>(node)];
      parents_[static_cast<size_t>(node)] = root;
      node = parent;
    }
    return root;
  }

  void Union(int64_t first, int64_t second) {
    const int64_t first_root = Find(first);
    const int64_t second_root = Find(second);
    if (first_root == second_root) {
      return;
    }
    const std::string component_uid = std::min(
        component_uids_[static_cast<size_t>(first_root)],
        component_uids_[static_cast<size_t>(second_root)]);
    parents_[static_cast<size_t>(first_root)] = second_root;
    component_uids_[static_cast<size_t>(second_root)] = component_uid;
  }

  int64_t AddEdge(int64_t first, int64_t second) {
    if (first == second) {
      throw std::invalid_argument("active track graph self edge is invalid");
    }
    const uint64_t low = static_cast<uint64_t>(std::min(first, second));
    const uint64_t high = static_cast<uint64_t>(std::max(first, second));
    const uint64_t edge = (low << 32) | high;
    if (!edges_.insert(edge).second) {
      return 0;
    }
    Union(first, second);
    return 1;
  }

  std::unordered_map<std::string, int64_t> node_by_uid_;
  std::vector<std::string> uid_by_node_;
  std::vector<int64_t> parents_;
  std::vector<std::string> component_uids_;
  std::vector<bool> alive_;
  std::vector<int64_t> tensor_rows_;
  std::vector<int64_t> free_tensor_rows_;
  int64_t next_tensor_row_ = 0;
  std::unordered_set<uint64_t> edges_;
  int64_t live_node_count_ = 0;
};

} // namespace

py::dict SelectActiveTrackCandidatesWrapper(
    py::array_t<int64_t, py::array::c_style> integer_metrics,
    py::array_t<double, py::array::c_style> float_metrics,
    const std::vector<std::string> &component_uids,
    int64_t minimum_track_views, double minimum_parallax_diagonals,
    int64_t maximum_active_tracks, int64_t maximum_tracks_per_grid_cell,
    int native_thread_count) {
  if (integer_metrics.ndim() != 3 || integer_metrics.shape(2) != 5 ||
      float_metrics.ndim() != 3 || float_metrics.shape(2) != 2 ||
      integer_metrics.shape(0) != float_metrics.shape(0) ||
      integer_metrics.shape(1) != float_metrics.shape(1) ||
      integer_metrics.shape(0) !=
          static_cast<int64_t>(component_uids.size())) {
    throw std::invalid_argument("active track metric arrays differ in shape");
  }
  if (minimum_track_views < 1 || minimum_parallax_diagonals < 0.0 ||
      maximum_active_tracks < 1 || maximum_tracks_per_grid_cell < 1 ||
      native_thread_count < 1) {
    throw std::invalid_argument("active track selection limits are invalid");
  }
  ActiveTrackSelectionBatch result;
  {
    py::gil_scoped_release release;
    result = SelectActiveTrackCandidates(
        integer_metrics.data(), float_metrics.data(), component_uids,
        integer_metrics.shape(0), integer_metrics.shape(1),
        minimum_track_views, minimum_parallax_diagonals,
        maximum_active_tracks, maximum_tracks_per_grid_cell,
        native_thread_count);
  }
  py::dict report;
  report["selectedComponentIndexes"] =
      VecToArray1D(std::move(result.selected_component_indexes));
  report["selectedOffsets"] =
      VecToArray1D(std::move(result.selected_offsets));
  report["candidateCounts"] =
      VecToArray1D(std::move(result.candidate_counts));
  report["bridgeCounts"] =
      VecToArray1D(std::move(result.bridge_counts));
  report["selectedTrackUidsSha256"] =
      std::move(result.selected_uids_sha256);
  report["bridgeTrackUidsSha256"] =
      std::move(result.bridge_uids_sha256);
  report["insufficientViewCounts"] =
      VecToArray1D(std::move(result.insufficient_view_counts));
  report["insufficientParallaxCounts"] =
      VecToArray1D(std::move(result.insufficient_parallax_counts));
  report["bucketOffsets"] =
      VecToArray1D(std::move(result.bucket_offsets));
  report["bucketTimeBins"] =
      VecToArray1D(std::move(result.bucket_time_bins));
  report["bucketRows"] = VecToArray1D(std::move(result.bucket_rows));
  report["bucketColumns"] =
      VecToArray1D(std::move(result.bucket_columns));
  report["bucketCounts"] = VecToArray1D(std::move(result.bucket_counts));
  return report;
}

void BindActiveTrackGraph(py::module_ &module) {
  py::class_<ActiveTrackGraph>(module, "ActiveTrackGraph")
      .def(py::init<>())
      .def("add_nodes", &ActiveTrackGraph::AddNodes, py::arg("uids"))
      .def("add_nodes_with_rows", &ActiveTrackGraph::AddNodesWithRows,
           py::arg("uids"))
      .def("add_edges", &ActiveTrackGraph::AddEdges,
           py::arg("first_uids"), py::arg("second_uids"))
      .def("add_star_edges", &ActiveTrackGraph::AddStarEdges,
           py::arg("track_indexes"), py::arg("view_indexes"),
           py::arg("resolved_uids"))
      .def("component_uids", &ActiveTrackGraph::ComponentUids,
           py::arg("uids"))
      .def("group_components", &ActiveTrackGraph::GroupComponents,
           py::arg("uids"))
      .def("group_component_rows", &ActiveTrackGraph::GroupComponentRows,
           py::arg("uids"), py::arg("frame_ids"))
      .def("remove_nodes", &ActiveTrackGraph::RemoveNodes,
           py::arg("uids"))
      .def("remove_nodes_with_rows", &ActiveTrackGraph::RemoveNodesWithRows,
           py::arg("uids"))
      .def_property_readonly("node_count", &ActiveTrackGraph::NodeCount)
      .def_property_readonly("edge_count", &ActiveTrackGraph::EdgeCount);
  module.def(
      "select_active_track_candidates", &SelectActiveTrackCandidatesWrapper,
      py::arg("integer_metrics"), py::arg("float_metrics"),
      py::arg("component_uids"), py::arg("minimum_track_views"),
      py::arg("minimum_parallax_diagonals"),
      py::arg("maximum_active_tracks"),
      py::arg("maximum_tracks_per_grid_cell"),
      py::arg("native_thread_count"),
      "Select active tracks with deterministic native OpenMP workers.");
}

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

std::vector<std::string> BatchObservationUidsWrapper(
    const std::string &prediction_uid,
    py::array_t<int64_t, py::array::c_style> track_indexes,
    py::array_t<int64_t, py::array::c_style> view_indexes,
    const std::vector<std::string> &frame_uids) {
  if (track_indexes.size() != view_indexes.size() ||
      track_indexes.size() != static_cast<int64_t>(frame_uids.size())) {
    throw std::invalid_argument("observation UID arrays differ in length");
  }
  std::vector<std::string> result;
  {
    py::gil_scoped_release release;
    result = BatchObservationUids(prediction_uid, track_indexes.data(),
                                  view_indexes.data(), frame_uids,
                                  track_indexes.size());
  }
  return result;
}
