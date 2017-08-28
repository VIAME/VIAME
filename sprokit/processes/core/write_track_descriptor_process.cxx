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
 * \brief Implementation for write_track_descriptor_set process
 */

#include "write_track_descriptor_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/write_track_descriptor_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "",
  "Name of the track descriptor set file to write." );
create_config_trait( writer, std::string , "",
  "Block name for algorithm parameters. "
  "e.g. writer:type would be used to specify the algorithm type." );

//--------------------------------------------------------------------------------
// Private implementation class
class write_track_descriptor_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::write_track_descriptor_set_sptr m_writer;
}; // end priv class


// ===============================================================================

write_track_descriptor_process
::write_track_descriptor_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new write_track_descriptor_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


write_track_descriptor_process
::~write_track_descriptor_process()
{
}


// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_configure()
{
  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );
  if ( d->m_file_name.empty() )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Required file name not specified." );
  }

  // Get algo conrig entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if ( ! algo::write_track_descriptor_set::check_nested_algo_configuration( "writer", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::write_track_descriptor_set::set_nested_algo_configuration( "writer", algo_config, d->m_writer);
  if ( ! d->m_writer )
  {
    throw sprokit::invalid_configuration_exception( name(),
             "Unable to create writer." );
  }
}


// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_init()
{
  d->m_writer->open( d->m_file_name ); // throws
}


// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_step()
{
  std::string file_name;

  // image name is optional
  if ( has_input_port_edge_using_trait( image_file_name ) )
  {
    file_name = grab_from_port_using_trait( image_file_name );
  }

  kwiver::vital::track_descriptor_set_sptr input
    = grab_from_port_using_trait( track_descriptor_set );

  d->m_writer->write_set( input );
}


// -------------------------------------------------------------------------------
void write_track_descriptor_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image_file_name, optional );
  declare_input_port_using_trait( track_descriptor_set, required );
}


// -------------------------------------------------------------------------------
void write_track_descriptor_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( writer );
}


// ===============================================================================
write_track_descriptor_process::priv
::priv()
{
}


write_track_descriptor_process::priv
::~priv()
{
}

} // end namespace
