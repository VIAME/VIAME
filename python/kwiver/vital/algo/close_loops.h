#ifndef KWIVER_VITAL_PYTHON_CLOSE_LOOPS_H_
#define KWIVER_VITAL_PYTHON_CLOSE_LOOPS_H_

#include <pybind11/pybind11.h>

namespace py = pybind11;

void close_loops(py::module &m);
#endif
