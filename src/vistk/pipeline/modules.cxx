/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "modules.h"

#if defined(_WIN32) || defined(_WIN64)
#include <vistk/pipeline/module-paths.h>
#endif
#include "utils.h"

#include <vistk/pipeline_util/path.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstddef>

/**
 * \file modules.cxx
 *
 * \brief Implementation of module loading.
 */

namespace vistk
{

namespace
{

#if defined(_WIN32) || defined(_WIN64)
typedef HMODULE library_t;
typedef FARPROC function_t;
#else
typedef void* library_t;
typedef void* function_t;
#endif
typedef void (*load_module_t)();
typedef path_t::string_type module_path_t;
typedef std::vector<module_path_t> module_paths_t;
typedef std::string lib_suffix_t;
typedef std::string function_name_t;

}

static void look_in_directory(module_path_t const& directory);
static void load_from_module(module_path_t const& path);
static bool is_separator(module_path_t::value_type ch);

static function_name_t const process_function_name = function_name_t("register_processes");
static function_name_t const scheduler_function_name = function_name_t("register_schedulers");
static module_path_t const default_module_dirs = module_path_t(DEFAULT_MODULE_PATHS);
static envvar_name_t const vistk_module_envvar = envvar_name_t("VISTK_MODULE_PATH");
static lib_suffix_t const library_suffix = lib_suffix_t(LIBRARY_SUFFIX);

void
load_known_modules()
{
  module_paths_t module_dirs;

  envvar_value_t const extra_module_dirs = get_envvar(vistk_module_envvar);

  if (extra_module_dirs)
  {
    boost::split(module_dirs, *extra_module_dirs, is_separator, boost::token_compress_on);
  }

  module_paths_t module_dirs_tmp;

  boost::split(module_dirs_tmp, default_module_dirs, is_separator, boost::token_compress_on);

  module_dirs.insert(module_dirs.end(), module_dirs_tmp.begin(), module_dirs_tmp.end());

  BOOST_FOREACH (module_path_t const& module_dir, module_dirs)
  {
    look_in_directory(module_dir);

#ifdef USE_CONFIGURATION_SUBDIRECTORY
    module_path_t const subdir = module_dir +
#if defined(_WIN32) || defined(_WIN64)
      L"/" VISTK_CONFIGURATION_L;
#else
      "/" VISTK_CONFIGURATION;
#endif
    ;

    look_in_directory(subdir);
#endif
  }
}

void
look_in_directory(module_path_t const& directory)
{
  if (directory.empty())
  {
    return;
  }

  if (!boost::filesystem::exists(directory))
  {
    /// \todo Log error that path doesn't exist.
    return;
  }

  if (!boost::filesystem::is_directory(directory))
  {
    /// \todo Log error that path isn't a directory.
    return;
  }

  boost::system::error_code ec;
  boost::filesystem::directory_iterator module_dir_iter(directory, ec);

  while (module_dir_iter != boost::filesystem::directory_iterator())
  {
    boost::filesystem::directory_entry const ent = *module_dir_iter;

    ++module_dir_iter;

    if (!boost::ends_with(ent.path().native(), library_suffix))
    {
      continue;
    }

    if (ent.status().type() != boost::filesystem::regular_file)
    {
      /// \todo Log warning that we found a non-file matching path.
      continue;
    }

    load_from_module(ent.path().native());
  }
}

void
load_from_module(module_path_t const& path)
{
  library_t library = NULL;

#if defined(_WIN32) || defined(_WIN64)
  library = LoadLibraryW(path.c_str());
#else
  library = dlopen(path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
#endif

  if (!library)
  {
    /// \todo Log an error.

    return;
  }

  function_t process_function = NULL;
  function_t scheduler_function = NULL;

#if defined(_WIN32) || defined(_WIN64)
  process_function = GetProcAddress(library, process_function_name.c_str());
  scheduler_function = GetProcAddress(library, scheduler_function_name.c_str());
#else
  process_function = dlsym(library, process_function_name.c_str());
  scheduler_function = dlsym(library, scheduler_function_name.c_str());
#endif

#ifdef __GNUC__
  // See https://trac.osgeo.org/qgis/ticket/234#comment:17
  __extension__
#endif
  load_module_t const process_registrar = reinterpret_cast<load_module_t>(process_function);
#ifdef __GNUC__
  // See https://trac.osgeo.org/qgis/ticket/234#comment:17
  __extension__
#endif
  load_module_t const scheduler_registrar = reinterpret_cast<load_module_t>(scheduler_function);

  bool functions_found = false;

  if (process_registrar)
  {
    /// \todo Log info that we have loaded processes.

    (*process_registrar)();
    functions_found = true;
  }
  if (scheduler_registrar)
  {
    /// \todo Log info that we have loaded schedulers.

    (*scheduler_registrar)();
    functions_found = true;
  }

  if (!functions_found)
  {
#if defined(_WIN32) || defined(_WIN64)
    int const ret = FreeLibrary(library);

    if (!ret)
    {
      /// \todo Log the error.
    }
#else
    int const ret = dlclose(library);

    if (ret)
    {
      /// \todo Log the error.
    }
#endif
  }
}

bool
is_separator(module_path_t::value_type ch)
{
  module_path_t::value_type const separator =
#if defined(_WIN32) || defined(_WIN64)
    ';';
#else
    ':';
#endif

  return (ch == separator);
}

}
