#ifndef KWIVER_VITAL_PYTHON_BUNDLE_ADJUST_H_
#define KWIVER_VITAL_PYTHON_BUNDLE_ADJUST_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bundle_adjust(py::module &m);
#endif
