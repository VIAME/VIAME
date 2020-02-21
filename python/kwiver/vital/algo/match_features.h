#ifndef KWIVER_VITAL_PYTHON_MATCH_FEATURES_H_
#define KWIVER_VITAL_PYTHON_MATCH_FEATURES_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void match_features(py::module &m);
}
}
}

#endif
