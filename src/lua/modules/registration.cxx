/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <lua/helpers/lua_include.h>

#include <luabind/object.hpp>
#include <luabind/detail/call_function.hpp>

#include <vistk/pipeline/process_registry.h>
#include <vistk/pipeline/schedule_registry.h>
#include <vistk/pipeline/utils.h>

#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include <sstream>

using namespace vistk;

static envvar_name_t const lua_suppress_envvar = envvar_name_t("VISTK_NO_LUA_MODULES");

static envvar_name_t const lua_process_modules_envvar = envvar_name_t("VISTK_LUA_SCHEDULE_MODULES");
static envvar_name_t const lua_schedule_modules_envvar = envvar_name_t("VISTK_LUA_PROCESS_MODULES");

static std::string const standard_lua_process_module = "vistk.processes";
static std::string const standard_lua_schedule_module = "vistk.schedules";

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

  modules.push_back(standard_lua_process_module);

  // Load extra modules given via the environment.
  {
    envvar_value_t const lua_modules = get_envvar(lua_process_modules_envvar);

    if (lua_modules)
    {
      /// \bug Boost <= 1.47 boost::split *overwrites* destination.
      std::vector<std::string> modules_tmp;

      boost::split(modules_tmp, lua_modules, is_separator, boost::token_compress_on);

      modules.insert(modules.end(), modules.begin(), modules_tmp.end());
    }

    free_envvar(lua_modules);
  }

  static process_registry::module_t const base_module_name = process_registry::module_t("lua_process_module:");

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

  modules.push_back(standard_lua_schedule_module);

  // Load extra modules given via the environment.
  {
    envvar_value_t const lua_modules = get_envvar(lua_schedule_modules_envvar);

    if (lua_modules)
    {
      /// \bug Boost <= 1.47 boost::split *overwrites* destination.
      std::vector<std::string> modules_tmp;

      boost::split(modules_tmp, lua_modules, is_separator, boost::token_compress_on);

      modules.insert(modules.end(), modules.begin(), modules_tmp.end());
    }

    free_envvar(lua_modules);
  }

  static schedule_registry::module_t const base_module_name = schedule_registry::module_t("lua_schedule_module:");

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
  envvar_value_t const lua_suppress = get_envvar(lua_suppress_envvar);

  bool suppress_lua_modules = false;

  if (lua_suppress)
  {
    suppress_lua_modules = true;
  }

  free_envvar(lua_suppress);

  return suppress_lua_modules;
}

void
load_from_module(std::string const& module, std::string const& function)
{
  lua_State* L = lua_open();
  luaL_openlibs(L);

  std::string const function_name = module + ":" + function;

  try
  {
    luabind::call_function(L, "require", module.c_str());
    luabind::call_function(L, function_name.c_str());

    dtor_registry_t reg = dtor_registry::self();

    reg->register_dtor(boost::bind(lua_close, L));
  }
  catch (luabind::error&)
  {
    lua_close(L);
  }
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
