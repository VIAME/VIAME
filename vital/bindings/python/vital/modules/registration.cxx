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

#include <vital/modules/modules_python_export.h>

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include <pybind11/pybind11.h>
#if WIN32
#pragma warning (pop)
#endif

//#include <sprokit/pipeline/utils.h>
#include <vital/bindings/python/vital/util/pybind11.h>
#include <vital/bindings/python/vital/util/python_exceptions.h>
#include <vital/bindings/python/vital/util/python.h>

#include <vital/plugin_loader/plugin_loader.h>
#include <kwiversys/SystemTools.hxx>

#ifdef VITAL_LOAD_PYLIB_SYM
  #include <dlfcn.h>
#endif

// Undefine macros that will double expand in case an definition has a value
// something like: /usr/lib/x86_64-linux-gnu/libpython2.7.so
#ifdef linux
#define _orig_linux linux
#undef linux
#endif

// for getting the value of a macro as a string literal
#define MACRO_STR_VALUE(x) _TO_STRING0(x)
#define _TO_STRING0(x) _TO_STRING1(x)
#define _TO_STRING1(x) #x

namespace py = pybind11;

static void load();
static bool is_suppressed();
static void _load_python_library_symbols();


// ==================================================================
/**
 * @brief Python module loader.
 *
 * This function is called by the plugin loader when it is scanning
 * all plugins. It looks like a standard registration entry point for
 * a set or processes, but it activates the python interpreter and
 * causes it to call vital.modules.module_loader.load_python_modules()
 *
 * Also note that setting the environment variable
 * VITAL_NO_PYTHON_MODULES will suppress loading all python modules.
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

  _load_python_library_symbols();

  {
    kwiver::vital::python::gil_scoped_acquire acquire;
    (void)acquire;
    VITAL_PYTHON_IGNORE_EXCEPTION(load())
  }
}


// ------------------------------------------------------------------
/*
 * Uses environment variables and compiler definitions to determine where the
 * python shared library is and load its symbols.
 */
void _load_python_library_symbols()
{
  auto logger = kwiver::vital::get_logger("vital.python_modules");

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
    if (!handle) {
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


// ------------------------------------------------------------------
void
load()
{
  py::object const modules = py::module::import("vital.modules.module_loader");
  py::object const loader = modules.attr("load_python_modules");
  loader();
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


// Redefine values that we hacked away
#ifdef _orig_linux
#define linux _orig_linux
#undef _orig_linux
#endif
