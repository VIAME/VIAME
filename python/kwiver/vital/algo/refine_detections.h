#ifndef KWIVER_VITAL_PYTHON_REFINE_DETECTIONS_H_
#define KWIVER_VITAL_PYTHON_REFINE_DETECTIONS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void refine_detections(py::module &m);
#endif
