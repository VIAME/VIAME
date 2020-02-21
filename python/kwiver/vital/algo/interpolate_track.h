#ifndef KWIVER_VITAL_PYTHON_INTERPOLATE_TRACK_H_
#define KWIVER_VITAL_PYTHON_INTERPOLATE_TRACK_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void interpolate_track(py::module &m);
}
}
}

#endif
