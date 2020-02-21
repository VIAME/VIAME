#ifndef KWIVER_VITAL_PYTHON_INTIALIZE_CAMERAS_LANDMARKS_H_
#define KWIVER_VITAL_PYTHON_INTIALIZE_CAMERAS_LANDMARKS_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void initialize_cameras_landmarks(py::module &m);
}
}
}

#endif
