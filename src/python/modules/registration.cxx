/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/utils.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include <sstream>

#include <Python.h>

using namespace vistk;

static envvar_name_t const python_suppress_envvar = envvar_name_t("VISTK_NO_PYTHON_MODULES");

static envvar_name_t const python_process_modules_envvar = envvar_name_t("VISTK_PYTHON_SCHEDULE_MODULES");
static envvar_name_t const python_schedule_modules_envvar = envvar_name_t("VISTK_PYTHON_PROCESS_MODULES");

static std::string const standard_python_process_module = "vistk.processes";
static std::string const standard_python_schedule_module = "vistk.schedules";

static bool is_suppressed();
static void load_from_module(std::string const& module, std::string const& function);
static bool is_separator(char ch);

void
register_processes()
{
  if (is_suppressed())
  {
    return;
  }

  std::vector<std::string> modules;

  modules.push_back(standard_python_process_module);

  // Load extra modules given via the environment.
  {
    envvar_value_t const python_modules = get_envvar(python_process_modules_envvar);

    if (python_modules)
    {
      /// \bug Boost <= 1.47 boost::split *overwrites* destination.
      std::vector<std::string> modules_tmp;

      boost::split(modules_tmp, python_modules, is_separator, boost::token_compress_on);

      modules.insert(modules.end(), modules.begin(), modules_tmp.end());
    }

    free_envvar(python_modules);
  }

  Py_Initialize();

  static process_registry::module_t const base_module_name = process_registry::module_t("python_process_module:");

  process_registry_t const registry = process_registry::self();

  BOOST_FOREACH (std::string const& module, modules)
  {
    process_registry::module_t const module_name = base_module_name + process_registry::module_t(module);

    if (registry->is_module_loaded(module_name))
    {
      continue;
    }

    std::string const function = "register_processes";

    load_from_module(module, function);

    registry->mark_module_as_loaded(module_name);
  }
}

void
register_schedules()
{
  if (is_suppressed())
  {
    return;
  }

  std::vector<std::string> modules;

  modules.push_back(standard_python_schedule_module);

  // Load extra modules given via the environment.
  {
    envvar_value_t const python_modules = get_envvar(python_schedule_modules_envvar);

    if (python_modules)
    {
      /// \bug Boost <= 1.47 boost::split *overwrites* destination.
      std::vector<std::string> modules_tmp;

      boost::split(modules_tmp, python_modules, is_separator, boost::token_compress_on);

      modules.insert(modules.end(), modules.begin(), modules_tmp.end());
    }

    free_envvar(python_modules);
  }

  Py_Initialize();

  static schedule_registry::module_t const base_module_name = schedule_registry::module_t("python_schedule_module:");

  schedule_registry_t const registry = schedule_registry::self();

  BOOST_FOREACH (std::string const& module, modules)
  {
    schedule_registry::module_t const module_name = base_module_name + schedule_registry::module_t(module);

    if (registry->is_module_loaded(module_name))
    {
      continue;
    }

    std::string const function = "register_schedule";

    load_from_module(module, function);

    registry->mark_module_as_loaded(module_name);
  }
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

  if (suppress_python_modules)
  {
    return true;
  }

  return false;
}

void
load_from_module(std::string const& module, std::string const& function)
{
  std::stringstream sstr;

  sstr << "import " << module << std::endl;
  sstr << "if hasattr(" << module << ", \'" << function << "\'):" << std::endl;
  sstr << "    if callable(" << module << "." << function << "):" << std::endl;
  sstr << "        " << module << "." << function << "()" << std::endl;

  PyRun_SimpleString(sstr.str().c_str());
}

bool is_separator(char ch)
{
  char const separator =
#if defined(_WIN32) || defined(_WIN64)
    ';';
#else
    ':';
#endif

  return (ch == separator);
}
