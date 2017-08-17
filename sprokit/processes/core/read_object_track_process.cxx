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
 * \brief Implementation for read_object_track_set process
 */

#include "read_object_track_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/read_object_track_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

create_config_trait( file_name, std::string, "",
  "Name of the track descriptor set file to read." );
create_config_trait( reader, std::string , "",
  "Algorithm type to use as the reader." );

//--------------------------------------------------------------------------------
// Private implementation class
class read_object_track_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::read_object_track_set_sptr m_reader;
}; // end priv class


// ===============================================================================

read_object_track_process
::read_object_track_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new read_object_track_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


read_object_track_process
::~read_object_track_process()
{
}


// -------------------------------------------------------------------------------
void read_object_track_process
::_configure()
{
  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );

  if( d->m_file_name.empty() )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Required file name not specified." );
  }

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if(  algo::read_object_track_set::check_nested_algo_configuration(
         "reader",
         algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::read_object_track_set::set_nested_algo_configuration(
    "reader",
    algo_config,
    d->m_reader );

  if( ! d->m_reader )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Unable to create reader." );
  }
}


// -------------------------------------------------------------------------------
void read_object_track_process
::_init()
{
  d->m_reader->open( d->m_file_name ); // throws
}


// -------------------------------------------------------------------------------
void read_object_track_process
::_step()
{
  std::string image_name;
  kwiver::vital::object_track_set_sptr set;

  if( d->m_reader->read_set( set ) )
  {
    push_to_port_using_trait( object_track_set, set );
  }
  else
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );

    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( object_track_set, dat );
  }
}


// -------------------------------------------------------------------------------
void read_object_track_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  declare_output_port_using_trait( object_track_set, optional );
}


// -------------------------------------------------------------------------------
void read_object_track_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( reader );
}


// ===============================================================================
read_object_track_process::priv
::priv()
{
}


read_object_track_process::priv
::~priv()
{
}

} // end namespace
