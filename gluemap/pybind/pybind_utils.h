#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace py = pybind11;

// Transfer ownership of a vector to a 1D numpy array via capsule (zero-copy).
template <typename T> py::array_t<T> VecToArray1D(std::vector<T> vec) {
  auto *owner = new std::vector<T>(std::move(vec));
  py::capsule capsule(owner,
                      [](void *p) { delete static_cast<std::vector<T> *>(p); });
  return py::array_t<T>({static_cast<py::ssize_t>(owner->size())},
                        {static_cast<py::ssize_t>(sizeof(T))}, owner->data(),
                        capsule);
}
