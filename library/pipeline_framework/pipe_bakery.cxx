// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pipe_bakery.h"

#include "pipe_bakery_exception.h"
#include "bakery_base.h"

#include "pipeline_builder.h"
#include "pipe_declaration_types.h"

#include <viame/algorithm_framework/config/config_block.h>

#include <viame/pipeline_framework/pipeline.h>
#include <viame/pipeline_framework/process.h>
#include <viame/pipeline_framework/process_factory.h>

#include <memory>

/**
 * \file pipe_bakery.cxx
 *
 * \brief Implementation of baking a pipeline.
 */

namespace viame::pipeline {

namespace {

static viame::config_block_key_t const config_pipeline_key = viame::config_block_key_t( "_pipeline" );

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
    std::visit( bakery, b );
  }

  bakery_base::config_decls_t& configs = bakery.m_configs;

  // Convert config entries to global config.
  viame::config_block_sptr global_conf = bakery_base::extract_configuration_from_decls( configs );

  // Create pipeline.
  viame::config_block_sptr const pipeline_conf = global_conf->subblock_view( config_pipeline_key );

  pipe = std::make_shared< pipeline > ( pipeline_conf );

  // Create processes.
  {
    for( bakery_base::process_decl_t const & decl : bakery.m_processes )
    {
      process::name_t const& proc_name = decl.first;
      process::type_t const& proc_type = decl.second;
      viame::config_block_sptr const proc_conf = global_conf->subblock_view( proc_name );

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

// ------------------------------------------------------------------
viame::config_block_sptr
extract_configuration( pipe_blocks const& blocks )
{
  pipe_bakery bakery;

  for (auto b : blocks )
  {
    std::visit( bakery, b );
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
