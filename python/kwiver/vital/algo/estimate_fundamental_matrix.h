#ifndef KWIVER_VITAL_PYTHON_ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define KWIVER_VITAL_PYTHON_ESTIMATE_FUNDAMENTAL_MATRIX_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_fundamental_matrix(py::module &m);
}
}
}

#endif
