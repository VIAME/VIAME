// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/modules/module_helpers.h>

#include <python/kwiver/vital/util/python.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <kwiversys/SystemTools.hxx>

#include <pybind11/pybind11.h>

#ifdef VITAL_LOAD_PYLIB_SYM
  #include <dlfcn.h>
#endif

// Undefine macros that will double expand in case an definition has a value
// something like: /usr/lib/x86_64-linux-gnu/libpython2.7.so
#ifdef linux
#define _orig_linux linux
#undef linux
#endif

namespace py = pybind11;
/*
 * Check if a python interpretor exists and initialize one the python interpretor
 * does not exist
 */
void check_and_initialize_python_interpretor()
{
  // Check if a python interpreter already exists so we don't clobber sys.argv
  // (e.g. if sprokit is initialized from python)
  if (!Py_IsInitialized())
  {
    // Embed a python interpretter if one does not exist
    Py_Initialize();

    // Set Python interpeter attribute: sys.argv = []
    // parameters are: (argc, argv, updatepath)
    PySys_SetArgvEx(0, NULL, 0);
  }

  if (!PyEval_ThreadsInitialized())
  {
    {
      // Let pybind11 initialize threads and set up its internal data structures
      pybind11::detail::get_internals();
    }
    // Release the GIL
    PyEval_SaveThread();
  }
}

/*
 * Use environment variables and compiler definitions to determine where the
 * python shared library is and load its symbols.
 */
bool load_python_library_from_env()
{
  auto logger = kwiver::vital::get_logger("vital.python_modules");
  bool python_library_found = false;
  #ifdef VITAL_LOAD_PYLIB_SYM
    const char *env_pylib = kwiversys::SystemTools::GetEnv( "PYTHON_LIBRARY" );

    // cmake should provide this definition
    #ifdef PYTHON_LIBRARY
      const char *default_pylib = MACRO_STR_VALUE(PYTHON_LIBRARY);
    #else
      const char *default_pylib = NULL;
    #endif

    // First check if the PYTHON_LIBRARY environment variable is specified
    if( env_pylib )
    {
      LOG_DEBUG(logger, "Loading symbols from PYTHON_LIBRARY=" << env_pylib );
      void* handle = dlopen( env_pylib, RTLD_LAZY | RTLD_GLOBAL );
      if ( !handle ) {
        LOG_ERROR(logger, "Cannot load library: " << dlerror());
      }
      else
      {
        python_library_found = true;
      }
    }
    else if( default_pylib )
    {
      // If the PYTHON_LIBRARY environment variable is not specified, use the
      // CMAKE definition of PYTHON_LIBRARY instead.
      LOG_DEBUG(logger, "Loading symbols from default PYTHON_LIBRARY=" << default_pylib);
      void* handle = dlopen( default_pylib, RTLD_LAZY | RTLD_GLOBAL );
      if (!handle) {
        LOG_ERROR(logger, "Cannot load library: " << dlerror());
      }
      else
      {
        python_library_found = true;
      }
    }
    else
    {
      LOG_DEBUG(logger, "Unable to pre-load python symbols because " <<
      "PYTHON_LIBRARY is undefined.");
    }
    #else
      LOG_DEBUG(logger, "Not checking for python symbols");
  #endif
  return python_library_found;
}

bool load_python_library_from_interpretor(const std::string python_library_path)
{
  auto logger = kwiver::vital::get_logger("vital.python_modules");
  LOG_DEBUG(logger, "Loading symbols from PYTHON_LIBRARY=" << python_library_path.c_str() );
  bool python_library_found = false;
  #ifdef VITAL_LOAD_PYLIB_SYM
    void *handle = dlopen( python_library_path.c_str(), RTLD_LAZY | RTLD_GLOBAL );
    if (!handle) {
      LOG_ERROR(logger, "Cannot load library: " << dlerror());
    }
    else
    {
      python_library_found = true;
    }
  #endif
  return python_library_found;
}

/*
 * Determine python library through interpretor
 */
std::string
find_python_library()
{
  auto logger = kwiver::vital::get_logger("vital.python_modules");
  // Handle the case where the import fail
  try
  {
    py::object const module = py::module::import("kwiver.vital.util.find_python_library");
    py::object const python_library_path = module.attr("find_python_library")();
    return python_library_path.cast<std::string>();
  }
  catch (pybind11::error_already_set& e)
  {
    LOG_WARN(logger, "Failed to use python interpretor to find python library:\n" << e.what());
    return std::string();
  }
}

// Redefine values that we hacked away
#ifdef _orig_linux
#define linux _orig_linux
#undef _orig_linux
#endif
