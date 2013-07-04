/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#if defined(_WIN32) || defined(_WIN64)
#include <processes/clusters/cluster-paths.h>
#endif

#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>

#include <sprokit/pipeline/process_registry.h>
#include <sprokit/pipeline/process_registry_exception.h>
#include <sprokit/pipeline/utils.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>

#include <algorithm>

/**
 * \file clusters/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace sprokit;

namespace
{

#if defined(_WIN32) || defined(_WIN64)
typedef std::wstring cluster_path_t;
#else
typedef std::string cluster_path_t;
#endif

}

static cluster_path_t const default_include_dirs = cluster_path_t(DEFAULT_CLUSTER_PATHS);
static envvar_name_t const sprokit_include_envvar = envvar_name_t("SPROKIT_CLUSTER_PATH");
static std::string const pipe_suffix = std::string(".cluster");

static bool is_separator(cluster_path_t::value_type ch);

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("cluster_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  process::types_t const current_types = registry->types();
  process::types_t new_types;

  typedef path_t include_path_t;
  typedef std::vector<include_path_t> include_paths_t;

  include_paths_t include_dirs;

  // Build include directories.
  {
    include_paths_t include_dirs_tmp;

    envvar_value_t const extra_include_dirs = get_envvar(sprokit_include_envvar);

    if (extra_include_dirs)
    {
      boost::split(include_dirs_tmp, *extra_include_dirs, is_separator, boost::token_compress_on);

      include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
    }

    boost::split(include_dirs_tmp, default_include_dirs, is_separator, boost::token_compress_on);

    include_dirs.insert(include_dirs.end(), include_dirs_tmp.begin(), include_dirs_tmp.end());
  }

  BOOST_FOREACH (include_path_t const& include_dir, include_dirs)
  {
    if (!boost::filesystem::exists(include_dir))
    {
      /// \todo Log error that path doesn't exist.
      continue;
    }

    if (!boost::filesystem::is_directory(include_dir))
    {
      /// \todo Log error that path isn't a directory.
      continue;
    }

    boost::system::error_code ec;
    boost::filesystem::directory_iterator module_dir_iter(include_dir, ec);

    /// \todo Check ec.

    while (module_dir_iter != boost::filesystem::directory_iterator())
    {
      boost::filesystem::directory_entry const ent = *module_dir_iter;

      ++module_dir_iter;

      path_t const path = ent.path();
      path_t::string_type const& pstr = path.native();

      if (!boost::ends_with(pstr, pipe_suffix))
      {
        continue;
      }

      if (ent.status().type() != boost::filesystem::regular_file)
      {
        /// \todo Log warning that we found a non-file matching path.
        continue;
      }

      cluster_info_t info;

      try
      {
        info = bake_cluster_from_file(path);
      }
      catch (load_pipe_exception const& /*e*/)
      {
        /// \todo Handle exceptions.

        continue;
      }
      catch (pipe_bakery_exception const& /*e*/)
      {
        /// \todo Handle exceptions.

        continue;
      }

      if (info)
      {
        process::type_t const& type = info->type;
        process_registry::description_t const& description = info->description;
        process_ctor_t const& ctor = info->ctor;

        try
        {
          registry->register_process(type, description, ctor);
        }
        catch (process_type_already_exists_exception const& e)
        {
          /// \todo Print out exception.
        }
      }
    }
  }

  registry->mark_module_as_loaded(module_name);
}

bool
is_separator(cluster_path_t::value_type ch)
{
  cluster_path_t::value_type const separator =
#if defined(_WIN32) || defined(_WIN64)
    ';';
#else
    ':';
#endif

  return (ch == separator);
}
