// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_OBJECT_TRACK_SET_H_
#define KWIVER_PYTHON_ARROW_SERIALIZE_JSON_SERIALIZE_OBJECT_TRACK_SET_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace arrows {
namespace python {
namespace py = pybind11;

void serialize_object_track_set(py::module &m);
}
}
}
#endif
