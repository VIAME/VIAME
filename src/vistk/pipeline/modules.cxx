/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "modules.h"

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

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
typedef char const* envvar_name_t;
typedef char const* envvar_value_t;
typedef std::string module_path_t;
typedef std::vector<module_path_t> module_paths_t;
typedef std::string lib_suffix_t;
typedef std::string function_name_t;

}

static void load_from_module(module_path_t const path);
static bool is_separator(char ch);

static function_name_t const process_function_name = function_name_t("register_processes");
static function_name_t const schedule_function_name = function_name_t("register_schedules");
static envvar_name_t const vistk_module_envvar = envvar_name_t("VISTK_MODULE_PATH");
static lib_suffix_t const library_suffix = lib_suffix_t(
#if defined(_WIN32) || defined(_WIN64)
  ".dll"
#elif defined(__APPLE__)
  ".dylib"
#else
  ".so"
#endif
  );

void load_known_modules()
{
  module_paths_t module_dirs;

#ifdef VISTK_LIBRARY_OUTPUT_PATH
  module_dirs.push_back(VISTK_LIBRARY_OUTPUT_PATH);
#endif

#ifdef VISTK_MODULE_INSTALL_PATH
  module_dirs.push_back(VISTK_MODULE_INSTALL_PATH);
#endif

  envvar_value_t const extra_module_dirs = getenv(vistk_module_envvar);

  if (extra_module_dirs)
  {
    boost::split(module_dirs, extra_module_dirs, is_separator, boost::token_compress_on);
  }

  BOOST_FOREACH (module_path_t const& module_dir, module_dirs)
  {
    if (module_dir.empty())
    {
      continue;
    }

    if (!boost::filesystem::exists(module_dir))
    {
      /// \todo Log error that path doesn't exist.
      continue;
    }

    if (!boost::filesystem::is_directory(module_dir))
    {
      /// \todo Log error that path isn't a directory.
      continue;
    }

    boost::system::error_code ec;
    boost::filesystem::directory_iterator module_dir_iter(module_dir, ec);

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
}

void load_from_module(module_path_t const path)
{
  library_t library = NULL;

#if defined(_WIN32) || defined(_WIN64)
  {
    wchar_t path_wide[MB_CUR_MAX];
    mbstowcs(path_wide, path.c_str(), MB_CUR_MAX);
    library = LoadLibrary(path_wide);
  }
#else
  library = dlopen(path.c_str(), RTLD_LAZY);
#endif

  if (!library)
  {
    return;
  }

  function_t process_function = NULL;
  function_t schedule_function = NULL;

#if defined(_WIN32) || defined(_WIN64)
  {
    wchar_t function_name[MB_CUR_MAX];

    mbstowcs(function_name, process_function_name.c_str(), MB_CUR_MAX);
    process_function = GetProcAddress(library, function_name);

    mbstowcs(function_name, schedule_function_name.c_str(), MB_CUR_MAX);
    schedule_function = GetProcAddress(library, function_name);
  }
#else
  process_function = dlsym(library, process_function_name.c_str());
  schedule_function = dlsym(library, schedule_function_name.c_str());
#endif

  load_module_t process_registrar = reinterpret_cast<load_module_t>(process_function);
  load_module_t schedule_registrar = reinterpret_cast<load_module_t>(schedule_function);

  bool functions_found = false;

  if (process_registrar)
  {
    /// \todo Log info that we have loaded processes.

    (*process_registrar)();
    functions_found = true;
  }
  if (schedule_registrar)
  {
    /// \todo Log info that we have loaded schedules.

    (*schedule_registrar)();
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

} // end namespace vistk
