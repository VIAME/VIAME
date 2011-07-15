/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "modules.h"

#include <string>

#include <dlfcn.h>

namespace vistk
{

typedef void* library_t;
typedef void* function_t;
typedef void (*load_module_t)();
typedef char const* module_path_t;
typedef char const* const function_name_t;

static void load_from_module(module_path_t const path);

static function_name_t const edge_function_name = function_name_t("register_edges");
static function_name_t const pipeline_function_name = function_name_t("register_pipelines");
static function_name_t const process_function_name = function_name_t("register_processes");

void load_known_modules()
{
  /// \todo Populate path listing.
  module_path_t const module_paths[] =
    { NULL
    };

  module_path_t const* module_path = module_paths;

  while (*module_path)
  {
    /// \todo Search for .so, .dylib, .dll within the directory.
    load_from_module(*module_path);
    ++module_path;
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

  function_t edge_function = dlsym(library, edge_function_name);
  function_t pipeline_function = dlsym(library, pipeline_function_name);
  function_t process_function = dlsym(library, process_function_name);

  load_module_t edge_registrar = reinterpret_cast<load_module_t>(edge_function);
  load_module_t pipeline_registrar = reinterpret_cast<load_module_t>(pipeline_function);
  load_module_t process_registrar = reinterpret_cast<load_module_t>(process_function);

  if (edge_registrar)
  {
    (*edge_registrar)();
  }
  if (pipeline_registrar)
  {
    (*pipeline_registrar)();
  }
  if (process_registrar)
  {
    (*process_registrar)();
  }
}

} // end namespace vistk
