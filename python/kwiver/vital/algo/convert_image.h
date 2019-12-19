#ifndef KWIVER_VITAL_PYTHON_COVERT_IMAGE_H_
#define KWIVER_VITAL_PYTHON_COVERT_IMAGE_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void convert_image(py::module &m);
#endif
