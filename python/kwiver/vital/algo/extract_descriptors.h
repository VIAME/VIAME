#ifndef KWIVER_VITAL_PYTHON_EXTRACT_DESCRIPTORS_H_
#define KWIVER_VITAL_PYTHON_EXTRACT_DESCRIPTORS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void extract_descriptors(py::module &m);
#endif
