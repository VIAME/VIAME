// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipe_bakery.h"

#include "pipe_bakery_exception.h"
#include "bakery_base.h"
#include "cluster_bakery.h"
#include "cluster_creator.h"

#include "pipeline_builder.h"
#include "pipe_declaration_types.h"

#include <vital/config/config_block.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process.h>
#include <sprokit/pipeline/process_cluster.h>
#include <sprokit/pipeline/process_factory.h>

#include <memory>

/**
 * \file pipe_bakery.cxx
 *
 * \brief Implementation of baking a pipeline.
 */

namespace sprokit {

namespace {

static kwiver::vital::config_block_key_t const config_pipeline_key = kwiver::vital::config_block_key_t( "_pipeline" );

} // end anonymous

// ==================================================================
class pipe_bakery :
  public bakery_base
{
public:
  pipe_bakery();
  ~pipe_bakery();

  using bakery_base::operator();
};

// ------------------------------------------------------------------
pipeline_t
bake_pipe_blocks( pipe_blocks const& blocks )
{
  pipeline_t pipe;

  pipe_bakery bakery;

  // apply main visitor to collect
  for ( auto b : blocks )
  {
    kwiver::vital::visit( bakery, b );
  }

  bakery_base::config_decls_t& configs = bakery.m_configs;

  // Convert config entries to global config.
  kwiver::vital::config_block_sptr global_conf = bakery_base::extract_configuration_from_decls( configs );

  // Create pipeline.
  kwiver::vital::config_block_sptr const pipeline_conf = global_conf->subblock_view( config_pipeline_key );

  pipe = std::make_shared< pipeline > ( pipeline_conf );

  // Create processes.
  {
    for( bakery_base::process_decl_t const & decl : bakery.m_processes )
    {
      process::name_t const& proc_name = decl.first;
      process::type_t const& proc_type = decl.second;
      kwiver::vital::config_block_sptr const proc_conf = global_conf->subblock_view( proc_name );

      // Create process with its config block.
      process_t const proc = create_process( proc_type, proc_name, proc_conf );

      pipe->add_process( proc );
    }
  }

  // Make connections.
  {
    for( process::connection_t const & conn : bakery.m_connections )
    {
      process::port_addr_t const& up = conn.first;
      process::port_addr_t const& down = conn.second;

      process::name_t const& up_name = up.first;
      process::port_t const& up_port = up.second;
      process::name_t const& down_name = down.first;
      process::port_t const& down_port = down.second;

      pipe->connect( up_name, up_port, down_name, down_port );
    }
  }

  return pipe;
} // bake_pipe_blocks

// ============================================================================
cluster_info_t
bake_cluster_blocks( cluster_blocks const& blocks )
{
  auto bakery = std::make_shared< cluster_bakery >();

  for ( auto b : blocks )
  {
    kwiver::vital::visit( *bakery, b );
  }

  if ( bakery->m_processes.empty() )
  {
    VITAL_THROW( cluster_without_processes_exception );
  }

  cluster_bakery::opt_cluster_component_info_t const& opt_cluster = bakery->m_cluster;

  if ( ! opt_cluster )
  {
    VITAL_THROW( missing_cluster_block_exception );
  }

  cluster_bakery::cluster_component_info_t const& cluster = *opt_cluster;

  if ( cluster.m_inputs.empty() &&
       cluster.m_outputs.empty() )
  {
    VITAL_THROW( cluster_without_ports_exception );
  }

  process::type_t const& type = bakery->m_type;
  process::description_t const& description = bakery->m_description;

  // Bakery is copied into cluster_creator so it can be const.
  process_factory_func_t const ctor = cluster_creator( *bakery );

  cluster_info_t const info = std::make_shared< cluster_info > ( type, description, ctor );
  info->m_bakery = bakery;

  return info;
}

// ------------------------------------------------------------------
kwiver::vital::config_block_sptr
extract_configuration( pipe_blocks const& blocks )
{
  pipe_bakery bakery;

  for (auto b : blocks )
  {
    kwiver::vital::visit( bakery, b );
  }

  bakery_base::config_decls_t& configs = bakery.m_configs;

  return bakery_base::extract_configuration_from_decls( configs );
}

// ------------------------------------------------------------------
pipe_bakery
::pipe_bakery()
  : bakery_base()
{
}

pipe_bakery
::~pipe_bakery()
{
}

}
