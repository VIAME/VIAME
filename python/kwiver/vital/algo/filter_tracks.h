#ifndef KWIVER_VITAL_PYTHON_FILTER_TRACKS_H_
#define KWIVER_VITAL_PYTHON_FILTER_TRACKS_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void filter_tracks(py::module &m);
}
}
}

#endif
