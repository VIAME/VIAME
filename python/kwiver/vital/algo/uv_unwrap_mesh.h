// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_UV_UNWRAP_MESH_H_
#define KWIVER_VITAL_PYTHON_UV_UNWRAP_MESH_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void uv_unwrap_mesh(py::module &m);
}
}
}

#endif
