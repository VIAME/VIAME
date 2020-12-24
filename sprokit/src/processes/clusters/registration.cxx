// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file clusters/registration.cxx
 *
 * \brief Register processes for use.
 */

#include "processes_clusters_export.h"

#include <kwiversys/Directory.hxx>
#include <kwiversys/SystemTools.hxx>
#include <sprokit/pipeline/process_factory.h>
#include <sprokit/pipeline/process_registry_exception.h>
#include <sprokit/pipeline/utils.h>
#include <sprokit/pipeline_util/load_pipe_exception.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>
#include <sprokit/pipeline_util/pipeline_builder.h>
#include <vital/logger/logger.h>
#include <vital/util/tokenize.h>
#include <vital/kwiver-include-paths.h>

#include <algorithm>
#include <iostream>

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
  using kvpf = kwiver::vital::plugin_factory;

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
      catch (std::exception const& e)
      {
        LOG_ERROR( logger, "Caught unexpected exception: " << e.what() );
        continue;
      }

      if (info)
      {

        LOG_DEBUG( logger, "Registering a cluster. Name: " << info->type );

        try
        {
          // Add cluster to process registry with a specific factory function
          auto fact = vpm.add_factory( new sprokit::cluster_process_factory( info ) );
          fact->add_attribute( kvpf::PLUGIN_MODULE_NAME,  reg.module_name() )
            .add_attribute( kvpf::PLUGIN_ORGANIZATION, reg.organization() )
            .add_attribute( "cluster-file", pstr ) // indicate cluster and source file name
            ;
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
