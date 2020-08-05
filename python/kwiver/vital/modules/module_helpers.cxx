/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
 * check if the python shared library symbols have been loaded
 * From: https://stackoverflow.com/questions/50570223/dlopenrtld-noload-still-returns-not-null-after-dlclose
 */
bool is_python_library_loaded(const std::string& python_library_path)
{
  void *handle = dlopen(python_library_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if ( handle != nullptr )
  {
    dlclose(handle);
    return true;
  }
  return false;
}

/*
 * Use environment variables and compiler definitions to determine where the
 * python shared library is and load its symbols.
 */
void load_python_library_symbols(const std::string python_library_path)
{
  auto logger = kwiver::vital::get_logger("vital.python_modules");
  if (python_library_path.empty())
  {
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
    }
    else
    {
      LOG_DEBUG(logger, "Unable to pre-load python symbols because " <<
                        "PYTHON_LIBRARY is undefined.");
    }
    #else
      LOG_DEBUG(logger, "Not checking for python symbols");
    #endif
  }
  else
  {
    LOG_DEBUG(logger, "Loading symbols from PYTHON_LIBRARY=" << python_library_path.c_str() );
    void *handle = dlopen( python_library_path.c_str(), RTLD_LAZY | RTLD_GLOBAL );
    if (!handle) {
      LOG_ERROR(logger, "Cannot load library: " << dlerror());
    }
  }
}

/*
 * Determine python library through interpretor
 */
std::string
find_python_library()
{
  py::object const module = py::module::import("kwiver.vital.util.find_python_library");
  py::object const python_library_path = module.attr("find_python_library")();
  return python_library_path.cast<std::string>();
}


// Redefine values that we hacked away
#ifdef _orig_linux
#define linux _orig_linux
#undef _orig_linux
#endif
