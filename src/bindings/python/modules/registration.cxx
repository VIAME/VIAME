/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <sprokit/pipeline/utils.h>

#include <sprokit/python/util/python_exceptions.h>
#include <sprokit/python/util/python_gil.h>

#include <boost/python/import.hpp>

#include <Python.h>

using namespace boost::python;

static sprokit::envvar_name_t const python_suppress_envvar = sprokit::envvar_name_t("SPROKIT_NO_PYTHON_MODULES");

static void load();
static bool is_suppressed();

void
register_processes()
{
  if (is_suppressed())
  {
    return;
  }

  Py_Initialize();

  sprokit::python::python_gil const gil;

  (void)gil;

  SPROKIT_PYTHON_IGNORE_EXCEPTION(load())
}

void
load()
{
  object const modules = import("sprokit.modules.modules");
  object const loader = modules.attr("load_python_modules");

  loader();
}

bool
is_suppressed()
{
  sprokit::envvar_value_t const python_suppress = sprokit::get_envvar(python_suppress_envvar);

  bool suppress_python_modules = false;

  if (python_suppress)
  {
    suppress_python_modules = true;
  }

  return suppress_python_modules;
}
