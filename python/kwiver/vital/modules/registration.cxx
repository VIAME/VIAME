// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/modules/modules_python_export.h>
#include <python/kwiver/vital/modules/module_helpers.h>

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/util/python_exceptions.h>
#include <python/kwiver/vital/util/python.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <kwiversys/SystemTools.hxx>

#include <pybind11/stl.h>
#include <string>
#include <algorithm>
// ==================================================================
/**
 * @brief Python module loader.
 *
 * This function is called by the plugin loader when it is scanning
 * all plugins. It looks like a standard registration entry point for
 * a set or processes, but it activates the python interpreter and
 * causes it to call vital.modules.module_loader.load_python_modules().
 * Addtionally for the python package of kwiver it is used to register external
 * c++ plugins by specifying a search paths for the plugins
 * Also note that setting the environment variable
 * VITAL_NO_PYTHON_MODULES will suppress loading all python modules.
 */

namespace py = pybind11;

static void load_python_modules();
static bool is_suppressed();
static void load_additional_cpp_modules(kwiver::vital::plugin_loader& vpm);

// ==================================================================
/**
 * @brief Python module loader.
 *
 * This function is called by the plugin loader when it is scanning
 * all plugins. It looks like a standard registration entry point for
 * a set or processes, but it activates the python interpreter and
 * causes it to call vital.modules.module_loader.load_python_modules().
 * Addtionally for the python package of kwiver it is used to register external
 * c++ plugins by specifying a search paths for the plugins
 * Also note that setting the environment variable
 * VITAL_NO_PYTHON_MODULES will suppress loading all python modules and any cpp
 * modules that are advertised through entrypoints.
 */

extern "C"
MODULES_PYTHON_EXPORT
void
register_factories(kwiver::vital::plugin_loader& vpm)
{
  if (is_suppressed())
  {
    return;
  }

  static auto const module_name = std::string( "module_python" );
  auto logger = kwiver::vital::get_logger(module_name);
  if(vpm.is_module_loaded(module_name))
  {
    return;
  }
  check_and_initialize_python_interpretor();
  bool python_library_loaded = load_python_library_from_env();
  if (!python_library_loaded)
  {
    std::string python_library_path = "";
    {
      kwiver::vital::python::gil_scoped_acquire acquire;
      (void)acquire;
      python_library_path = find_python_library();
    }
    if(!python_library_path.empty())
    {
      python_library_loaded = load_python_library_from_interpretor(python_library_path);
    }
  }
  if (!python_library_loaded)
  {
      LOG_ERROR(logger, "Cannot load python library from interpretor or env");
  }
  // Load python modules
  {
    kwiver::vital::python::gil_scoped_acquire acquire;
    (void)acquire;
    VITAL_PYTHON_IGNORE_EXCEPTION(load_python_modules())
  }

  {
    kwiver::vital::python::gil_scoped_acquire acquire;
    (void)acquire;
    VITAL_PYTHON_IGNORE_EXCEPTION(load_additional_cpp_modules(vpm))
  }
  vpm.mark_module_as_loaded(module_name);
}

// ------------------------------------------------------------------
bool
is_suppressed()
{
  const char * python_suppress = kwiversys::SystemTools::GetEnv( "SPROKIT_NO_PYTHON_MODULES" );
  bool suppress_python_modules = false;

  if (python_suppress)
  {
    suppress_python_modules = true;
  }

  return suppress_python_modules;
}

// ------------------------------------------------------------------
void
load_python_modules()
{
  py::object const modules = py::module::import("kwiver.vital.modules.module_loader");
  py::object const loader = modules.attr("load_python_modules");
  loader();
}

// -------------------------------------------------------------------
void
load_additional_cpp_modules(kwiver::vital::plugin_loader& vpm)
{
  auto logger = kwiver::vital::get_logger("vital.load_additional_cpp_paths");
  py::object const modules =  py::module::import("kwiver.vital.util.entrypoint");
  py::object const get_cpp_paths_from_entrypoint = modules.attr("get_cpp_paths_from_entrypoint");
  py::object py_additional_paths = get_cpp_paths_from_entrypoint();
  auto additional_paths = py_additional_paths.cast<std::vector<std::string>>();
  auto current_search_paths = vpm.get_search_path();
  auto new_search_paths = std::vector<std::string>();
  for(auto& current_search_path : current_search_paths)
  {
      LOG_INFO(logger, "Current search path" + current_search_path);
  }

  for(auto& additional_path: additional_paths)
  {
    if(std::find(current_search_paths.begin(),
                 current_search_paths.end(),
                 additional_path) == current_search_paths.end())
    {
      new_search_paths.push_back(additional_path);
      LOG_INFO(logger, "new search path" + additional_path);
    }
  }
  vpm.load_plugins(new_search_paths);
}
