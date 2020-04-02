/*ckwg +29
 * Copyright 2017-2018, 2020 by Kitware, Inc.
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
 * \brief Implementation for write_object_track_set process
 */

#include "write_object_track_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/write_object_track_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "",
  "Name of the track descriptor set file to write." );

create_algorithm_name_config_trait( writer );

//--------------------------------------------------------------------------------
// Private implementation class
class write_object_track_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::write_object_track_set_sptr m_writer;
}; // end priv class


// ===============================================================================

write_object_track_process
::write_object_track_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new write_object_track_process::priv )
{
  make_ports();
  make_config();
  set_data_checking_level( check_sync );
}


write_object_track_process
::~write_object_track_process()
{
}


// -------------------------------------------------------------------------------
void write_object_track_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );
  if ( d->m_file_name.empty() )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Required file name not specified." );
  }

  // Get algo conrig entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if( ! algo::write_object_track_set::check_nested_algo_configuration_using_trait(
        writer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::write_object_track_set::set_nested_algo_configuration_using_trait(
    writer,
    algo_config,
    d->m_writer );

  if( ! d->m_writer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Unable to create writer." );
  }
}


// -------------------------------------------------------------------------------
void write_object_track_process
::_init()
{
  d->m_writer->open( d->m_file_name ); // throws
}


// -------------------------------------------------------------------------------
void write_object_track_process
::_step()
{
  auto const& port_info = peek_at_port_using_trait( object_track_set );
  if( port_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( object_track_set );
    d->m_writer->close();
    mark_process_as_complete();
    return;
  }

  auto const& input = grab_from_port_using_trait( object_track_set );
  auto const& ts = try_grab_from_port_using_trait( timestamp );
  auto const& file_name = try_grab_from_port_using_trait( image_file_name );

  {
    scoped_step_instrumentation();

    d->m_writer->write_set( input, ts, file_name );
  }
}


// -------------------------------------------------------------------------------
void write_object_track_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image_file_name, optional );
  declare_input_port_using_trait( object_track_set, required );
  declare_input_port_using_trait( timestamp, optional );
}


// -------------------------------------------------------------------------------
void write_object_track_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( writer );
}


// ===============================================================================
write_object_track_process::priv
::priv()
{
}


write_object_track_process::priv
::~priv()
{
}

} // end namespace
