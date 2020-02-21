#ifndef KWIVER_VITAL_PYTHON_READ_TRACK_DESCRIPTOR_SET_H_
#define KWIVER_VITAL_PYTHON_READ_TRACK_DESCRIPTOR_SET_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void read_track_descriptor_set(py::module &m);
}
}
}

#endif
