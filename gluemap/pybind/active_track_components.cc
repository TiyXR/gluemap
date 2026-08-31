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

  void RemoveNodes(const std::vector<std::string> &uids) {
    py::gil_scoped_release release;
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
  std::unordered_set<uint64_t> edges_;
  int64_t live_node_count_ = 0;
};

} // namespace

void BindActiveTrackGraph(py::module_ &module) {
  py::class_<ActiveTrackGraph>(module, "ActiveTrackGraph")
      .def(py::init<>())
      .def("add_nodes", &ActiveTrackGraph::AddNodes, py::arg("uids"))
      .def("add_edges", &ActiveTrackGraph::AddEdges,
           py::arg("first_uids"), py::arg("second_uids"))
      .def("add_star_edges", &ActiveTrackGraph::AddStarEdges,
           py::arg("track_indexes"), py::arg("view_indexes"),
           py::arg("resolved_uids"))
      .def("component_uids", &ActiveTrackGraph::ComponentUids,
           py::arg("uids"))
      .def("remove_nodes", &ActiveTrackGraph::RemoveNodes,
           py::arg("uids"))
      .def_property_readonly("node_count", &ActiveTrackGraph::NodeCount)
      .def_property_readonly("edge_count", &ActiveTrackGraph::EdgeCount);
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
