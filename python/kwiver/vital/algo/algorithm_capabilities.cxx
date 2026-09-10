// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/algorithm_capabilities.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace kv = kwiver::vital;

namespace kwiver {

namespace vital {

namespace python {

void
algorithm_capabilities( py::module& m )
{
  py::class_< kv::algorithm_capabilities >( m, "AlgorithmCapabilities" )
    .def( py::init<>() )
    .def(
      "has_capability", &kv::algorithm_capabilities::has_capability,
      py::arg( "name" ) )
    .def( "capability_list", &kv::algorithm_capabilities::capability_list )
    .def(
      "capability", &kv::algorithm_capabilities::capability,
      py::arg( "name" ) )
    .def(
      "set_capability", &kv::algorithm_capabilities::set_capability,
      py::arg( "name" ), py::arg( "val" ) );
}

} // namespace python

} // namespace vital

} // namespace kwiver
