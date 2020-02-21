#ifndef KWIVER_VITAL_PYTHON_TRAIN_DETECTOR_H_
#define KWIVER_VITAL_PYTHON_TRAIN_DETECTOR_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void train_detector(py::module &m);
}
}
}

#endif
