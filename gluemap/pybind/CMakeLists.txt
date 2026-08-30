# setup PYTHON_EXECUTABLE
cmake_minimum_required(VERSION 3.18)

project(pygluemap)
cmake_policy(SET CMP0057 NEW)  # Fix IN_LIST handling

set(CMAKE_BUILD_TYPE Release)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(
  pygluemap MODULE
  "active_track_components.cc"
  "bindings.cc"
  "track_selection.cc")

target_include_directories(pygluemap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
target_link_libraries(
  pygluemap PRIVATE
  pybind11::module
  Eigen3::Eigen
  ${CERES_LIBRARIES}
  Boost::headers
  OpenMP::OpenMP_CXX)
target_compile_definitions(pygluemap PRIVATE cxx_std_17)
if (TARGET Ceres::ceres_cuda_kernels)
    set(CERES_HAS_CUDA TRUE)
else()
    set(CERES_HAS_CUDA FALSE)
endif()
message(STATUS "Ceres CUDA support: ${CERES_HAS_CUDA}")

if (CERES_HAS_CUDA)
    target_compile_definitions(pygluemap PRIVATE CERES_HAS_CUDA)
endif()

# The immutable Linux package is installed at a declared path.  Let its
# builder provide that stable RPATH instead of baking a temporary build or
# bind-mount directory into the wheel.  Local editable builds keep the
# historical link-path behaviour when no explicit value is provided.
set(GLUEMAP_RUNTIME_RPATH "" CACHE STRING
    "Explicit runtime library search path for packaged pygluemap builds")
if (GLUEMAP_RUNTIME_RPATH)
    set_target_properties(pygluemap PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${GLUEMAP_RUNTIME_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH FALSE
        INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
    )
else()
    set_target_properties(pygluemap PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# scikit-build-core sets CMAKE_INSTALL_PREFIX to the wheel staging dir;
# combined with `wheel.install-dir = "/"` in pyproject.toml, this lands
# pygluemap at the top of site-packages (matching the legacy install).
install(TARGETS pygluemap DESTINATION .)
