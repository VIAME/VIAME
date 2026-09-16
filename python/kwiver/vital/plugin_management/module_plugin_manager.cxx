// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <viame/algorithm_framework/config/format_config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/test_interface/say.h>

namespace py = pybind11;
namespace kv = viame;

// ----------------------------------------------------------------------------
PYBIND11_MODULE( _plugin_management, m )
{
  py::enum_< kv::plugin_manager::plugin_type >( m, "PluginType" )
    .value( "PROCESSES", kv::plugin_manager::plugin_type::PROCESSES )
    .value( "ALGORITHMS", kv::plugin_manager::plugin_type::ALGORITHMS )
    .value( "APPLETS", kv::plugin_manager::plugin_type::APPLETS )
    .value( "EXPLORER", kv::plugin_manager::plugin_type::EXPLORER )
    .value( "OTHERS", kv::plugin_manager::plugin_type::OTHERS )
    .value( "LEGACY", kv::plugin_manager::plugin_type::LEGACY )
    .value( "DEFAULT", kv::plugin_manager::plugin_type::DEFAULT )
    .value( "ALL", kv::plugin_manager::plugin_type::ALL )
    .export_values();

// -------------------------------------------------------------------------
// Using unique_ptr rather than shared_ptr since destructor of plugin_manager
// is protected
  py::class_< kv::plugin_manager, std::unique_ptr< kv::plugin_manager,
    py::nodelete > >(
    m, "PluginManager",
    // doc-string
    "Main plugin manager for all kwiver components."
    )
  // Use lambda function for now so we don't have to deal bitflags type
    .def(
    "load_all_plugins", []( kv::plugin_manager& self ){
      return self.load_all_plugins();
    },
    py::doc( "Registers every factory VIAME was built with." )
    )
    .def(
      "reload_all_plugins", &kv::plugin_manager::reload_all_plugins,
      py::doc( "Clears the factory list and reloads plugins." )
    )
    .def(
      "impl_names", &kv::plugin_manager::_impl_names,
      py::doc( "Get list of plugin implementation names for an interface." )
    );

// -------------------------------------------------------------------------
  m.def(
    "plugin_manager_instance", &kv::plugin_manager::instance,
    py::return_value_policy::reference,
    // doc-string
    "Returns the plugin manager singleton"
  );

// -------------------------------------------------------------------------
  py::class_< kv::implementation_factory_by_name< kv::say > >(
    m, "SayFactory",
    // doc-string
    "Factory for say interface implementations."
  )
    .def( py::init<>() )
    .def(
      "create",  &kv::implementation_factory_by_name< kv::say >::create,
      py::arg( "value" ), py::arg( "cb" ),
      py::doc(
        "Create an instance of the 'value' implemetation of the say interface." )
    );
}
