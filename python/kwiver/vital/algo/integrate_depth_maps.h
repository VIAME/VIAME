#ifndef KWIVER_VITAL_PYTHON_INTEGRATE_DEPTH_MAPS_H_
#define KWIVER_VITAL_PYTHON_INTEGRATE_DEPTH_MAPS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void integrate_depth_maps(py::module &m);
#endif
