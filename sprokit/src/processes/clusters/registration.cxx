/*ckwg +29
 * Copyright 2012-2018 by Kitware, Inc.
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

/**
 * \file clusters/registration.cxx
 *
 * \brief Register processes for use.
 */

#include "processes_clusters_export.h"

#include <processes/clusters/cluster-paths.h>

#include <vital/logger/logger.h>
#include <vital/util/tokenize.h>

#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>

#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process_registry_exception.h>
#include <sprokit/pipeline/utils.h>

#include <algorithm>
#include <kwiversys/SystemTools.hxx>
#include <kwiversys/Directory.hxx>

using namespace sprokit;

static std::string const default_include_dirs = std::string(DEFAULT_CLUSTER_PATHS);
static std::string const sprokit_include_envvar = std::string("SPROKIT_CLUSTER_PATH");
static std::string const cluster_suffix = std::string(".cluster");
static const std::string path_separator( 1, PATH_SEPARATOR_CHAR );

// ------------------------------------------------------------------
/**
 * \brief Cluster loader.
 *
 * This function loads and instantiates cluster processes.
 */
extern "C"
PROCESSES_CLUSTERS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  process_registrar reg( vpm, "cluster_processes" );

  static kwiver::vital::logger_handle_t logger = kwiver::vital::get_logger( "sprokit.register_cluster" );

  // See if clusters have already been loaded
  if ( reg.is_module_loaded() )
  {
    return;
  }

  kwiver::vital::path_list_t include_dirs;

  // Build include directories.
  kwiversys::SystemTools::GetPath( include_dirs, sprokit_include_envvar.c_str() );
  kwiver::vital::tokenize( default_include_dirs, include_dirs, path_separator, kwiver::vital::TokenizeTrimEmpty );

  for ( const kwiver::vital::path_t& include_dir : include_dirs)
  {
    // log file
    LOG_DEBUG( logger, "Loading clusters from directory: " << include_dir );
    if ( ! kwiversys::SystemTools::FileExists( include_dir) )
    {
      LOG_DEBUG( logger, "Path not found loading clusters: " << include_dir );
      continue;
    }

    if ( ! kwiversys::SystemTools::FileIsDirectory(include_dir) )
    {
      LOG_WARN( logger, "Path not directory loading clusters: " << include_dir );
      continue;
    }


    kwiversys::Directory dir;
    dir.Load( include_dir );
    unsigned long num_files = dir.GetNumberOfFiles();

    for (unsigned long i = 0; i < num_files; ++i )
    {
      std::string pstr = dir.GetPath();
      pstr += "/" + std::string( dir.GetFile( i ) );

      if ( kwiversys::SystemTools::GetFilenameLastExtension( pstr ) != cluster_suffix )
      {
        continue;
      }

      // log loading file
      LOG_DEBUG( logger, "Loading cluster from file: " << pstr );

      // Check that we're looking a file
      if ( kwiversys::SystemTools::FileIsDirectory( pstr ) )
      {
        LOG_WARN( logger, "Found non-file loading clusters: " << pstr );
        continue;
      }

      cluster_info_t info;

      try
      {
        // Compile cluster specification
        pipeline_builder builder;

        builder.load_cluster( pstr );
        info = builder.cluster_info();
      }
      catch (load_pipe_exception const& e)
      {
        LOG_WARN( logger, "Exception caught loading cluster: " << e.what() );
        continue;
      }
      catch (pipe_bakery_exception const& e)
      {
        LOG_WARN( logger, "Exception caught processing cluster definition: " << e.what() );
        continue;
      }

      if (info)
      {
        process::type_t const& type = info->type;
        std::string const& description = info->description;
        process_factory_func_t const& ctor = info->ctor;

        try
        {
          // Add cluster to process registry with a specific factory function
          auto fact = vpm.add_factory( new sprokit::cpp_process_factory( type, typeid( sprokit::process ).name(), ctor ) );
          fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, description );

          // Indicate this is a cluster and add source file name
          fact->add_attribute( "sprokit.cluster", pstr );
        }
        catch (kwiver::vital::plugin_already_exists const& e)
        {
          LOG_WARN( logger, "Exception caught loading cluster: " << e.what() );
        }
      }
    }
  }

  reg.mark_module_as_loaded();
}
