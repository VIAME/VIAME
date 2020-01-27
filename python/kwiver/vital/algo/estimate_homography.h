#ifndef KWIVER_VITAL_PYTHON_ESTIMATE_HOMOGRAPHY_H_
#define KWIVER_VITAL_PYTHON_ESTIMATE_HOMOGRAPHY_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void estimate_homography(py::module &m);
#endif
