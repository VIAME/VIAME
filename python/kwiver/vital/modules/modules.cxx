// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/plugin/plugin_manager.h>

#include <pybind11/pybind11.h>

#include <algorithm>

/**
 * \file module_loader.cxx
 *
 * \brief Python bindings for module loading.
 */

namespace py = pybind11;

namespace kwiver {

namespace vital {

namespace python {

// Both of these used to ask python where the plugin directory was and hand
// the answer to the manager before doing anything. P8-T03 links the plugins
// in, so there is no directory and nothing to tell it.
void
load_known_modules()
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();
}

bool
is_module_loaded( std::string module_name )
{
  return kwiver::vital::plugin_manager::instance().is_module_loaded(
    module_name );
}

} // namespace python

} // namespace vital

} // namespace kwiver

PYBIND11_MODULE( modules, m )
{
  m.def(
    "load_known_modules", &kwiver::vital::python::load_known_modules,
    "Loads modules to populate the process and scheduler registries." );
  m.def(
    "is_module_loaded", &kwiver::vital::python::is_module_loaded,
    "Check if a module has been loaded" );
}
