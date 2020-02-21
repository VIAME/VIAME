#ifndef KWIVER_VITAL_PYTHON_WRITE_TRACK_DESCRIPTOR_SET_H_
#define KWIVER_VITAL_PYTHON_WRITE_TRACK_DESCRIPTOR_SET_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void write_track_descriptor_set(py::module &m);
#endif
}
}
}
