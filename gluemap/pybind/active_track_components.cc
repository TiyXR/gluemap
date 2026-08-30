#include "pybind_utils.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
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
