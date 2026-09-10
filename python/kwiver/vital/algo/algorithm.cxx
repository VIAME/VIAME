// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <python/kwiver/vital/algo/algorithm.h>
#include <python/kwiver/vital/algo/algorithm_trampoline.txx>
#include <viame/algorithm_framework/algo/algorithm.h>

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

void
algorithm( py::module& m )
{
  py::module::import( "kwiver.vital.config" );

  // import the module containing the binding of the pluggable class so we can
  // use it as a parent here.
  // Having kwiver::vital::pluggable in the hierarcy is improtant because the
  // plugin discovery process
  // registers only class that are a subclass of Pluggable (i.e.
  // kwiver::vital::pluggable).
  py::object const mod_pluggable = py::module::import( "kwiver.vital.plugins" );

  py::class_< kwiver::vital::algorithm,
    std::shared_ptr< kwiver::vital::algorithm >,
    kwiver::vital::pluggable,
    algorithm_trampoline<> >( m, "_algorithm" )
    .def_property(
    "impl_name", &kwiver::vital::algorithm::impl_name,
    &kwiver::vital::algorithm::set_impl_name )
    .def_static( "interface_name", &kwiver::vital::algorithm::interface_name )
    .def( "get_configuration", &kwiver::vital::algorithm::get_configuration )
    .def(
      "set_configuration", &kwiver::vital::algorithm::set_configuration,
      py::arg( "config" ) )
    .def(
      "check_configuration",
      &kwiver::vital::algorithm::check_configuration, py::arg( "config" ) );
}

} // namespace python

} // namespace vital

} // namespace kwiver
