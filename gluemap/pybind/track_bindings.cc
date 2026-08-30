#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

PYBIND11_MODULE(pygluemap_tracks, module) {
  module.def(
      "compute_connected_components", &ComputeConnectedComponentsWrapper,
      py::arg("node_count"), py::arg("edge_first"), py::arg("edge_second"),
      "Compute integer graph component labels with GIL-free OpenMP workers.");
  module.def(
      "batch_spatial_intern", &BatchSpatialInternWrapper,
      py::arg("existing_frames"), py::arg("existing_x"),
      py::arg("existing_y"), py::arg("existing_uids"),
      py::arg("incoming_frames"), py::arg("incoming_x"),
      py::arg("incoming_y"), py::arg("incoming_uids"), py::arg("radius"),
      "Match one incoming Star one-to-one against existing observations.");
  module.def(
      "batch_observation_uids", &BatchObservationUidsWrapper,
      py::arg("prediction_uid"), py::arg("track_indexes"),
      py::arg("view_indexes"), py::arg("frame_uids"),
      "Compute stable observation SHA-256 identities with OpenMP workers.");
}
