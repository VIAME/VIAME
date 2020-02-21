#ifndef KWIVER_VITAL_PYTHON_ESTIMATE_PNP_H_
#define KWIVER_VITAL_PYTHON_ESTIMATE_PNP_H_

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

void estimate_pnp(py::module &m);

}
}
}

#endif
