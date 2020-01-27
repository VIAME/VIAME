#ifndef KWIVER_VITAL_PYTHON_ESTIMATE_CANONICAL_TRANSFORM_H_
#define KWIVER_VITAL_PYTHON_ESTIMATE_CANONICAL_TRANSFORM_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void estimate_canonical_transform(py::module &m);
#endif
