#ifndef KWIVER_VITAL_PYTHON_COMPUTE_STEREO_DEPTH_MAP_H_
#define KWIVER_VITAL_PYTHON_COMPUTE_STEREO_DEPTH_MAP_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void compute_stereo_depth_map(py::module &m);

#endif
