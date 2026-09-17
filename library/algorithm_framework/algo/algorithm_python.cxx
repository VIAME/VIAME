// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include "algorithm_python.h"
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/algorithm.h>

namespace py = pybind11;

namespace viame {

namespace python {

void
algorithm( py::module& m )
{
  py::module::import( "viame.config" );

  // import the module containing the binding of the pluggable class so we can
  // use it as a parent here.
  // Having viame::pluggable in the hierarcy is improtant because the
  // plugin discovery process
  // registers only class that are a subclass of Pluggable (i.e.
  // viame::pluggable).
  py::object const mod_pluggable = py::module::import( "viame.plugins" );

  py::class_< viame::algorithm,
    std::shared_ptr< viame::algorithm >,
    viame::pluggable,
    algorithm_trampoline<> >( m, "_algorithm" )
    .def_property(
    "impl_name", &viame::algorithm::impl_name,
    &viame::algorithm::set_impl_name )
    .def_static( "interface_name", &viame::algorithm::interface_name )
    .def( "get_configuration", &viame::algorithm::get_configuration )
    .def(
      "set_configuration", &viame::algorithm::set_configuration,
      py::arg( "config" ) )
    .def(
      "check_configuration",
      &viame::algorithm::check_configuration, py::arg( "config" ) );
}

} // namespace python

} // namespace viame
