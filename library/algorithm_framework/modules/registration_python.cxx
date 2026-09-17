// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "module_helpers_python.h"
#include <viame/algorithm_framework/modules/modules_python_export.h>

#include <viame/algorithm_framework/util/file_system.h>
#include <pybind11/pybind11.h>
#include <viame/algorithm_framework/util/python.h>
#include <viame/algorithm_framework/util/python_exceptions.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include <algorithm>
#include <exception>
#include <pybind11/stl.h>
#include <string>
// ==================================================================

/**
 * @brief Python module loader.
 *
 * This library's registration function looks like any other -- the static
 * registry calls it with the rest -- but what it registers is python: it
 * activates the interpreter and has it call
 * viame_algorithm_framework.modules.module_loader.load_python_modules().
 * Setting the environment variable VITAL_NO_PYTHON_MODULES suppresses it.
 */

namespace py = pybind11;

static void load_python_modules();
static bool is_suppressed();

// ==================================================================

/**
 * @brief Python module loader.
 *
 * This is a standard registration entry point in shape, but what it
 * registers is python: it activates the interpreter and has it call
 * viame_algorithm_framework.modules.module_loader.load_python_modules().
 * Setting the environment variable VITAL_NO_PYTHON_MODULES suppresses it.
 */

// Python plugin discovery is best effort: a host with a broken or missing
// python environment should lose the python plugins, not die. This entry point
// is called through a function pointer from the registry, so anything that
// escapes it unwinds through an extern "C" boundary and reaches std::terminate
// -- which aborts every process that loads this plugin, kwiver's own tools
// included. The body already ignores python exceptions; catch everything else
// here so the same is true of the C++ ones.
static void register_factories_impl( viame::registry& vpm );

extern "C"
MODULES_PYTHON_EXPORT
void
register_factories( viame::registry& vpm )
{
  auto logger = viame::get_logger( "viame_algorithm_framework.python_modules" );
  try
  {
    register_factories_impl( vpm );
  }
  catch( std::exception const& e )
  {
    LOG_ERROR(
      logger,
      "Python plugin registration failed, continuing without the python "
      "plugins: " << e.what() );
  }
  catch( ... )
  {
    LOG_ERROR(
      logger,
      "Python plugin registration failed with an unrecognized exception, "
      "continuing without the python plugins" );
  }
}

void
register_factories_impl( viame::registry& vpm )
{
  if( is_suppressed() )
  {
    return;
  }

  static auto const module_name = std::string( "module_python" );
  auto logger = viame::get_logger( module_name );
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }
  if( !check_and_initialize_python_interpretor() )
  {
    // No Python interpreter could be initialized (for example a C++ host with
    // no Python environment configured).  Skip Python plugin discovery rather
    // than leaving the process in a half-initialized state.
    LOG_WARN(
      logger,
      "No Python interpreter available; skipping Python plugin discovery" );
    vpm.mark_module_as_loaded( module_name );
    return;
  }

  bool python_library_loaded = load_python_library_from_env();
  if( !python_library_loaded )
  {
    std::string python_library_path = "";
    {
      pybind11::gil_scoped_acquire acquire;
      ( void ) acquire;
      python_library_path = find_python_library();
    }
    if( !python_library_path.empty() )
    {
      python_library_loaded =
        load_python_library_from_interpretor( python_library_path );
    }
  }
  if( !python_library_loaded )
  {
    LOG_ERROR(logger, "Cannot load python library from interpretor or env" );
  }
  // Load python modules
  {
    pybind11::gil_scoped_acquire acquire;
    ( void ) acquire;
    VITAL_PYTHON_IGNORE_EXCEPTION( load_python_modules() )
  }
  vpm.mark_module_as_loaded( module_name );
}

// ------------------------------------------------------------------
bool
is_suppressed()
{
  const char* python_suppress =
    viame::get_env_renamed( "VIAME_NO_PYTHON_MODULES",
                            "SPROKIT_NO_PYTHON_MODULES" );
  bool suppress_python_modules = false;

  if( python_suppress )
  {
    suppress_python_modules = true;
  }

  return suppress_python_modules;
}

// ------------------------------------------------------------------
void
load_python_modules()
{
  py::object const modules =
    py::module::import( "viame.modules.module_loader" );
  py::object const loader = modules.attr( "load_python_modules" );
  loader();
}

