/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <python/helpers/exceptions.h>

#include <vistk/pipeline/utils.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/import.hpp>

#include <Python.h>

using namespace boost::python;
using namespace vistk;

static envvar_name_t const python_suppress_envvar = envvar_name_t("VISTK_NO_PYTHON_MODULES");

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

  vistk::python::python_gil const gil;

  (void)gil;

  HANDLE_PYTHON_EXCEPTION_IGNORE(load())
}

void
load()
{
  object const modules = import("vistk.modules.modules");
  object const loader = modules.attr("load_python_modules");

  loader();
}

bool
is_suppressed()
{
  envvar_value_t const python_suppress = get_envvar(python_suppress_envvar);

  bool suppress_python_modules = false;

  if (python_suppress)
  {
    suppress_python_modules = true;
  }

  free_envvar(python_suppress);

  return suppress_python_modules;
}
