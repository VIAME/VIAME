/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
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

#include <vital/logger/logger.h>

#include <algorithm>
//+ #include <iostream>

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


// ------------------------------------------------------------------
void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("cluster_processes");
  static kwiver::vital::logger_handle_t s_logger = kwiver::vital::get_logger( "sprokit:register_cluster" );

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
    // log file
    LOG_DEBUG( s_logger, "Loading clusters from directory: " << include_dir );
    if (!boost::filesystem::exists(include_dir))
    {
      LOG_WARN( s_logger, "Path not found loading clusters: " << include_dir );
      continue;
    }

    if (!boost::filesystem::is_directory(include_dir))
    {
      LOG_WARN( s_logger, "Path not directory loading clusters: " << include_dir );
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

      // log loading file
      LOG_DEBUG( s_logger, "Loading cluster from file: " << pstr );

      if (ent.status().type() != boost::filesystem::regular_file)
      {
        LOG_WARN( s_logger, "Found non-file loading clusters: " << pstr );
        continue;
      }

      cluster_info_t info;

      try
      {
        info = bake_cluster_from_file(path);
      }
      catch (load_pipe_exception const& e)
      {
        LOG_ERROR( s_logger, "Exception caught loading cluster: " << e.what() );
        continue;
      }
      catch (pipe_bakery_exception const& e)
      {
        LOG_ERROR( s_logger, "Exception caught loading cluster: " << e.what() );
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
          LOG_ERROR( s_logger, "Exception caught loading cluster: " << e.what() );
        }
      }
    }
  }

  registry->mark_module_as_loaded(module_name);
}


// ------------------------------------------------------------------
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
