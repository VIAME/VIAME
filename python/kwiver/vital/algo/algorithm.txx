// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.
#ifndef KWIVER_VITAL_PYTHON_ALGORITHM_TXX_
#define KWIVER_VITAL_PYTHON_ALGORITHM_TXX_
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

// register template specializations for methods in vital/algo/algorithm.txx

namespace py = pybind11;

namespace kwiver::vital::python {

template < typename INTERFACE, class... Args >
void
register_algorithm( py::class_< Args... >& c )
{
  c.def_static(
    "create_algorithm",
    &kwiver::vital::create_algorithm< INTERFACE >,
    py::arg( "implementation_name" ) );

  c.def_static(
    "create",
    &kwiver::vital::create_algorithm< INTERFACE >,
    py::arg( "implementation_name" ) );

  c.def_static(
    "has_algorithm_impl_name",
    &kwiver::vital::has_algorithm_impl_name< INTERFACE >,
    py::arg( "implementation_name" ) );

  c.def_static(
    "registered_names",
    &kwiver::vital::registered_names< INTERFACE > );

  c.def_static(
    "get_nested_algo_configuration",
    &kwiver::vital::get_nested_algo_configuration< INTERFACE >,
    py::arg( "name" ), py::arg( "config" ), py::arg( "nested_algo" ) );
  c.def_static(
    "set_nested_algo_configuration",
    [](std::string const& name, kwiver::vital::config_block_sptr config){
      std::shared_ptr< INTERFACE > nested_algo;
      kwiver::vital::set_nested_algo_configuration< INTERFACE >(
        name, config,
        nested_algo );
      return nested_algo;
    }, py::arg( "name" ), py::arg( "config" ) );
  c.def_static(
    "check_nested_algo_configuration",
    &kwiver::vital::check_nested_algo_configuration< INTERFACE >,
    py::arg( "name" ), py::arg( "config" ) );
}

} // namespace kwiver::vital::python

#endif
