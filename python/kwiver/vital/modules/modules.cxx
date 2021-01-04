// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/plugin_loader/plugin_manager.h>

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

const std::string get_initial_plugin_path()
{
  py::object const initial_plugin_path_module =
    py::module::import("kwiver.vital.util.initial_plugin_path");
  std::string const initial_plugin_path =
    initial_plugin_path_module.attr("get_initial_plugin_path")().cast<std::string>();
  return initial_plugin_path;
}

void add_external_plugin_paths()
{
  py::object const entrypoint_module =
    py::module::import("kwiver.vital.util.entrypoint");
  entrypoint_module.attr("add_entrypoint_paths_to_env")();
}

//@todo Alternative is to provide C bindings for the plugin manager.
void load_known_modules()
{
  const std::string initial_plugin_path =
                kwiver::vital::python::get_initial_plugin_path();

  kwiver::vital::plugin_manager::instance().add_search_path( initial_plugin_path );
  kwiver::vital::plugin_manager::instance().load_all_plugins();
}

bool is_module_loaded(std::string module_name)
{
  const std::string initial_plugin_path =
                kwiver::vital::python::get_initial_plugin_path();

  kwiver::vital::plugin_manager::instance().add_search_path( initial_plugin_path );
  return kwiver::vital::plugin_manager::instance().is_module_loaded(module_name);
}
}
}
}

PYBIND11_MODULE(modules, m)
{
  m.def("load_known_modules", &kwiver::vital::python::load_known_modules
    , "Loads modules to populate the process and scheduler registries.");
  m.def("is_module_loaded", &kwiver::vital::python::is_module_loaded,
      "Check if a module has been loaded");
}
