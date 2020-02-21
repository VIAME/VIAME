#ifndef KWIVER_VITAL_PYTHON_IMAGE_FILTER_H_
#define KWIVER_VITAL_PYTHON_IMAGE_FILTER_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void image_filter(py::module &m);
}
}
}

#endif
