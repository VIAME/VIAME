/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#include <sprokit/src/bindings/python/modules/modules_python_export.h>

#include <boost/python/import.hpp>

#include <sprokit/pipeline/utils.h>
#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>
#include <sprokit/python/util/python.h>

#include <vital/plugin_loader/plugin_loader.h>
#include <kwiversys/SystemTools.hxx>

#ifdef SPROKIT_LOAD_PYLIB_SYM
  #include <dlfcn.h>
#endif

using namespace boost::python;

static void load();
static bool is_suppressed();


// ==================================================================
/**
 * @brief Python module loader.
 *
 * This function is called by the plugin loader when it is scanning
 * all plugins. It looks like a standard registration entry point for
 * a set or processes, but it activates the python interpreter and
 * causes it to call sprokit.module.modules.load_python_modules()
 *
 * Also note that setting the environment variable
 * SPROKIT_NO_PYTHON_MODULES will suppress loading all python modules.
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

  Py_Initialize();

#ifdef SPROKIT_LOAD_PYLIB_SYM
  const char *pylib = kwiversys::SystemTools::GetEnv( "PYTHON_LIBRARY" );

  if( pylib )
  {
    dlopen( pylib, RTLD_LAZY | RTLD_GLOBAL );
  }
#endif

  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_IGNORE_EXCEPTION(load())
}


// ------------------------------------------------------------------
void
load()
{
  object const modules = import("sprokit.modules.modules");
  object const loader = modules.attr("load_python_modules");

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
