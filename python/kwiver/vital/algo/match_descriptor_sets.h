#ifndef KWIVER_VITAL_PYTHON_MATCH_DESCRIPTOR_SETS_H_
#define KWIVER_VITAL_PYTHON_MATCH_DESCRIPTOR_SETS_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void match_descriptor_sets(py::module &m);
}
}
}

#endif
