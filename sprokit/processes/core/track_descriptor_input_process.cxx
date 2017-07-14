/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation for track_descriptor_set_input process
 */

#include "track_descriptor_input_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/track_descriptor_set_input.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "", "Name of the track descriptor set file to read." );
create_config_trait( reader, std::string , "", "Algorithm type to use as the reader." );

//----------------------------------------------------------------
// Private implementation class
class track_descriptor_input_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::track_descriptor_set_input_sptr m_reader;
}; // end priv class


// ================================================================

track_descriptor_input_process
::track_descriptor_input_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new track_descriptor_input_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


track_descriptor_input_process
::~track_descriptor_input_process()
{
}


// ----------------------------------------------------------------
void track_descriptor_input_process
::_configure()
{
  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );
  if ( d->m_file_name.empty() )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Required file name not specified." );
  }

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if ( ! algo::track_descriptor_set_input::check_nested_algo_configuration( "reader", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::track_descriptor_set_input::set_nested_algo_configuration( "reader", algo_config, d->m_reader);
  if ( ! d->m_reader )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Unable to create reader." );
  }
}


// ----------------------------------------------------------------
void track_descriptor_input_process
::_init()
{
  d->m_reader->open( d->m_file_name ); // throws
}


// ----------------------------------------------------------------
void track_descriptor_input_process
::_step()
{
  std::string image_name;
  kwiver::vital::track_descriptor_set_sptr set;
  if ( d->m_reader->read_set( set, image_name ) )
  {
    push_to_port_using_trait( image_file_name, image_name );
    push_to_port_using_trait( track_descriptor_set, set );
  }
  else
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );

    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( track_descriptor_set, dat );
    push_datum_to_port_using_trait( image_file_name, dat );
  }
}


// ----------------------------------------------------------------
void track_descriptor_input_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  declare_output_port_using_trait( image_file_name, optional );
  declare_output_port_using_trait( track_descriptor_set, optional );
}


// ----------------------------------------------------------------
void track_descriptor_input_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( reader );
}


// ================================================================
track_descriptor_input_process::priv
::priv()
{
}


track_descriptor_input_process::priv
::~priv()
{
}

} // end namespace
