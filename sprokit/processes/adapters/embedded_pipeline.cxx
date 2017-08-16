/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \file
 * \brief Implementation for embedded_pipeline implementation.
 */

#include "embedded_pipeline.h"

#include <vital/config/config_block.h>
#include <vital/vital_foreach.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/scheduler.h>
#include <sprokit/pipeline/scheduler_factory.h>

#include <sprokit/processes/adapters/input_adapter.h>
#include <sprokit/processes/adapters/input_adapter_process.h>

#include <sprokit/processes/adapters/output_adapter.h>
#include <sprokit/processes/adapters/output_adapter_process.h>

#include <kwiversys/SystemTools.hxx>

#include <sstream>
#include <stdexcept>


namespace {

static kwiver::vital::config_block_key_t const scheduler_block = kwiver::vital::config_block_key_t("_scheduler");

} // end


namespace kwiver {

typedef kwiversys::SystemTools ST;

// ----------------------------------------------------------------
class embedded_pipeline::priv
{
public:
  // -- CONSTRUCTORS --
  priv()
    : m_logger( kwiver::vital::get_logger( "embedded_pipeline" ))
    , m_at_end( false )
    , m_pipeline_started( false )
    , m_input_adapter_connected( false )
    , m_output_adapter_connected (false )
  {
  }


  ~priv()
  {
    // If the pipeline has been started, wait until it has completed
    // before freeing storage. May have to do more here to deal with a
    // still-running pipeline.
    if ( m_pipeline_started )
    {
      try
      {
        m_scheduler->stop();
      }
      catch ( ... ) { }
    }
  }

  bool connect_input_adapter();
  bool connect_output_adapter();

//---------------------------
  vital::logger_handle_t m_logger;
  bool m_at_end;
  bool m_pipeline_started;
  bool m_input_adapter_connected;
  bool m_output_adapter_connected;

  kwiver::input_adapter m_input_adapter;
  kwiver::output_adapter m_output_adapter;

  sprokit::pipeline_t m_pipeline;
  kwiver::vital::config_block_sptr m_pipe_config;
  kwiver::vital::config_block_sptr m_scheduler_config;
  sprokit::scheduler_t m_scheduler;

}; // end class embedded_pipeline::priv


// ==================================================================
embedded_pipeline
::embedded_pipeline()
  : m_logger( kwiver::vital::get_logger( "sprokit.embedded_pipeline" ) )
  , m_priv( new priv() )
{
  // load processes
  kwiver::vital::plugin_manager::instance().load_all_plugins();

}


embedded_pipeline
::~embedded_pipeline()
{
}


// ------------------------------------------------------------------
void
embedded_pipeline
::build_pipeline( std::istream& istr, std::string const& def_dir )
{
  // create a pipeline
  sprokit::pipeline_builder builder;

  std::string cur_file( def_dir );
  if ( def_dir.empty() )
  {
    cur_file = ST::GetCurrentWorkingDirectory();
  }

  builder.load_pipeline( istr, cur_file + "/in-stream" );

  // build pipeline
  m_priv->m_pipeline = builder.pipeline();
  m_priv->m_pipe_config = builder.config();

  if ( ! m_priv->m_pipeline)
  {
    throw std::runtime_error( "Unable to bake pipeline" );
  }

  // perform setup operation on pipeline and get it ready to run
  // This throws many exceptions
  try
  {
    m_priv->m_pipeline->setup_pipeline();
  }
  catch( sprokit::pipeline_exception const& e)
  {
    std::stringstream str;
    str << "Error setting up pipeline: " << e.what();
    throw std::runtime_error( str.str() );
  }

  if ( ! connect_input_adapter() || ! connect_output_adapter() )
  {
    throw std::runtime_error( "Unable to connect to input and/or output adapter processes");
  }

  //
  // Setup scheduler
  //
  // Determine if new scheduler type has been specified in the config
  sprokit::scheduler::type_t scheduler_type =
    m_priv->m_pipe_config->get_value(
      scheduler_block + kwiver::vital::config_block::block_sep + "type",  // key string
      sprokit::scheduler_factory::default_type ); // default value

  // Get config sub block based on selected scheduler type from the main config
  m_priv->m_scheduler_config = m_priv->m_pipe_config->subblock(scheduler_block +
                          kwiver::vital::config_block::block_sep + scheduler_type);

  m_priv->m_scheduler = sprokit::create_scheduler(scheduler_type,
                                                  m_priv->m_pipeline,
                                                  m_priv->m_scheduler_config);

  if ( ! m_priv->m_scheduler)
  {
    throw std::runtime_error( "Unable to create scheduler" );
  }
}


// ------------------------------------------------------------------
void
embedded_pipeline
::send( kwiver::adapter::adapter_data_set_t ads )
{
  if ( ! input_adapter_connected() )
  {
    throw std::runtime_error( "Input adapter not connected." );
  }

  m_priv->m_input_adapter.send( ads );
}


// ------------------------------------------------------------------
void
embedded_pipeline
::send_end_of_input()
{
  if ( ! input_adapter_connected() )
  {
    throw std::runtime_error( "Input adapter not connected." );
  }

  auto ds = kwiver::adapter::adapter_data_set::create( kwiver::adapter::adapter_data_set::end_of_input );
  this->send( ds );
}


// ------------------------------------------------------------------
  kwiver::adapter::adapter_data_set_t
embedded_pipeline
::receive()
{
  if ( ! output_adapter_connected() )
  {
    throw std::runtime_error( "Output adapter not connected." );
  }

  if ( m_priv->m_at_end )
  {
    LOG_ERROR( m_logger, "receive() called after end_of_data processed. "
               << "Probable deadlock." );
  }

  auto ads =  m_priv->m_output_adapter.receive();
  m_priv->m_at_end = ads->is_end_of_data();
  return ads;
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::full() const
{
  if ( ! input_adapter_connected() )
  {
    throw std::runtime_error( "Input adapter not connected." );
  }

  return m_priv->m_input_adapter.full();
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::empty() const
{
  if ( ! output_adapter_connected() )
  {
    throw std::runtime_error( "Output adapter not connected." );
  }

  return m_priv->m_output_adapter.empty();
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::at_end() const
{
  return m_priv->m_at_end;
}


// ------------------------------------------------------------------
void
embedded_pipeline
::start()
{
  m_priv->m_scheduler->start();
}


// ------------------------------------------------------------------
void
embedded_pipeline
::wait()
{
  m_priv->m_scheduler->wait();
}


// ------------------------------------------------------------------
sprokit::process::ports_t
embedded_pipeline
::input_port_names() const
{
  if ( ! input_adapter_connected() )
  {
    throw std::runtime_error( "Input adapter not connected." );
  }

  return m_priv->m_input_adapter.port_list();
}


// ------------------------------------------------------------------
sprokit::process::ports_t
embedded_pipeline
::output_port_names() const
{
  if ( ! output_adapter_connected() )
  {
    throw std::runtime_error( "Output adapter not connected." );
  }

  return m_priv->m_output_adapter.port_list();
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::connect_input_adapter()
{
  return m_priv->connect_input_adapter();
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::connect_output_adapter()
{
  return m_priv->connect_output_adapter();
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::input_adapter_connected() const
{
  return m_priv->m_input_adapter_connected;
}


// ------------------------------------------------------------------
bool
embedded_pipeline
::output_adapter_connected() const
{
  return m_priv->m_output_adapter_connected;
}


// ==================================================================
bool
embedded_pipeline::priv::
connect_input_adapter()
{
  auto names = m_pipeline->process_names();
  VITAL_FOREACH( auto n, names )
  {
    auto proc = m_pipeline->process_by_name( n );
    if ( proc->type() == "input_adapter" )
    {
      m_input_adapter.connect( proc->name(), m_pipeline );
      m_input_adapter_connected = true;
      return true;
    }
  }
  return false;
}


// ------------------------------------------------------------------
bool
embedded_pipeline::priv::
connect_output_adapter()
{
  auto names = m_pipeline->process_names();
  VITAL_FOREACH( auto n, names )
  {
    auto proc = m_pipeline->process_by_name( n );
    if (proc->type() == "output_adapter" )
    {
      m_output_adapter.connect( proc->name(), m_pipeline );
      m_output_adapter_connected = true;
      return true;
    }
  }
  return false;
}


} // end namespace kwiver
