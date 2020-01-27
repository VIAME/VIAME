#ifndef KWIVER_VITAL_PYTHON_UUID_FACTORY_H_
#define KWIVER_VITAL_PYTHON_UUID_FACTORY_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void uuid_factory(py::module &m);
#endif
