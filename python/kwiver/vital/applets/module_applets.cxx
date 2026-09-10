// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace kwiver {

namespace tools {

namespace python {

void kwiver_applet_binding( py::module& m );

} // namespace python

} // namespace tools

} // namespace kwiver

PYBIND11_MODULE( _applets, m )
{
  kwiver::tools::python::kwiver_applet_binding( m );
}
