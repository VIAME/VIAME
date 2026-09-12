// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_PYTHON_VIAME_ALGORITHM_FRAMEWORK_ALGO_IMAGE_OBJECT_DETECTOR_H
#define KWIVER_PYTHON_VIAME_ALGORITHM_FRAMEWORK_ALGO_IMAGE_OBJECT_DETECTOR_H

#include <pybind11/pybind11.h>

namespace kwiver::vital::python {
namespace py = pybind11;

void image_object_detector(py::module& m);
}
#endif
