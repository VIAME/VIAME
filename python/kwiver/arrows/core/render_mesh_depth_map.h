// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_ARROW_CORE_RENDER_MESH_DEPTH_MAP_H_
#define KWIVER_PYTHON_ARROW_CORE_RENDER_MESH_DEPTH_MAP_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void render_mesh_depth_map(py::module &m);

#endif
