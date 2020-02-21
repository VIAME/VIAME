#ifndef KWIVER_VITAL_PYTHON_TRACK_FEATURES_H_
#define KWIVER_VITAL_PYTHON_TRACK_FEATURES_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void track_features(py::module &m);
}
}
}

#endif
