#ifndef KWIVER_VITAL_PYTHON_KEYFRAME_SELECTION_H_
#define KWIVER_VITAL_PYTHON_KEYFRAME_SELECTION_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void keyframe_selection(py::module &m);
}
}
}

#endif
