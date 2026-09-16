// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_VIAME_ALGORITHM_FRAMEWORK_ALGO_DETECTED_OBJECT_FILTER_H
#define KWIVER_PYTHON_VIAME_ALGORITHM_FRAMEWORK_ALGO_DETECTED_OBJECT_FILTER_H

#include <pybind11/pybind11.h>

namespace viame::python {
namespace py = pybind11;

void detected_object_filter(py::module& m);
}
#endif
