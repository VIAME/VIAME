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

#include <dlfcn.h>

namespace vistk
{

typedef void* library_t;
typedef void* function_t;
typedef void (*load_module_t)();
typedef char const* envvar_name_t;
typedef char const* envvar_value_t;
typedef std::string module_path_t;
typedef std::vector<module_path_t> module_paths_t;
typedef std::string lib_suffix_t;
typedef std::string function_name_t;

static void load_from_module(module_path_t const path);
static bool is_separator(char ch);

static function_name_t const edge_function_name = function_name_t("register_edges");
static function_name_t const pipeline_function_name = function_name_t("register_pipelines");
static function_name_t const process_function_name = function_name_t("register_processes");
static envvar_name_t const vistk_module_envvar = envvar_name_t("VISTK_MODULE_PATH");
static lib_suffix_t const library_suffix = lib_suffix_t(".so");

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
  /// \todo Support more than POSIX. Use kwsys?
  library_t library = dlopen(path, RTLD_LAZY);

  if (!library)
  {
    return;
  }

  function_t edge_function = dlsym(library, edge_function_name.c_str());
  function_t pipeline_function = dlsym(library, pipeline_function_name.c_str());
  function_t process_function = dlsym(library, process_function_name.c_str());

  load_module_t edge_registrar = reinterpret_cast<load_module_t>(edge_function);
  load_module_t pipeline_registrar = reinterpret_cast<load_module_t>(pipeline_function);
  load_module_t process_registrar = reinterpret_cast<load_module_t>(process_function);

  bool functions_found = false;

  if (edge_registrar)
  {
    (*edge_registrar)();
    functions_found = true;
  }
  if (pipeline_registrar)
  {
    (*pipeline_registrar)();
    functions_found = true;
  }
  if (process_registrar)
  {
    (*process_registrar)();
    functions_found = true;
  }

  if (!functions_found)
  {
    int const ret = dlclose(library);

    if (ret)
    {
      /// \todo Log the error.
    }
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
