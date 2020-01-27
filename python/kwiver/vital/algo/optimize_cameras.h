#ifndef KWIVER_VITAL_PYTHON_OPTIMIZE_CAMERAS_H_
#define KWIVER_VITAL_PYTHON_OPTIMIZE_CAMERAS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void optimize_cameras(py::module &m);
#endif
