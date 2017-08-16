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

#include "burnout_track_descriptors.h"

#include <string>
#include <sstream>
#include <exception>

#include <vital/exceptions.h>
#include <vital/vital_foreach.h>

#include <descriptors/online_descriptor_computer_process.h>


namespace kwiver {
namespace arrows {
namespace burnout {

#define DUMMY_OUTPUT 1

// ==================================================================================
class burnout_track_descriptors::priv
{
public:
  priv()
    : m_config_file( "burnout_descriptors.conf" )
    , m_process( "descriptor_computer" )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_config_file;

  vidtk::online_descriptor_computer_process< vxl_byte > m_process;
  vital::logger_handle_t m_logger;
};


// ==================================================================================
burnout_track_descriptors
::burnout_track_descriptors()
  : d( new priv() )
{

}


burnout_track_descriptors
::~burnout_track_descriptors()
{}


// ----------------------------------------------------------------------------------
vital::config_block_sptr
burnout_track_descriptors
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "config", d->m_config_file,  "Name of config file." );

  return config;
}


// ----------------------------------------------------------------------------------
void
burnout_track_descriptors
::set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );
  d->m_config_file = config->get_value< std::string >( "config" );

#ifndef DUMMY_OUTPUT
  vidtk::config_block vidtk_config = d->m_process.params();
  vidtk_config.parse( d->m_config_file );

  if( !d->m_process.set_params( vidtk_config ) )
  {
    std::string reason = "Failed to set pipeline parameters";
    throw vital::algorithm_configuration_exception( type_name(), impl_name(), reason );
  }

  if( !d->m_process.initialize() )
  {
    std::string reason = "Failed to initialize pipeline";
    throw vital::algorithm_configuration_exception( type_name(), impl_name(), reason );
  }
#endif
}


// ----------------------------------------------------------------------------------
bool
burnout_track_descriptors
::check_configuration( vital::config_block_sptr config ) const
{
  std::string config_fn = config->get_value< std::string >( "config" );

  if( config_fn.empty() )
  {
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------------
vital::track_descriptor_set_sptr
burnout_track_descriptors
::compute( vital::image_container_sptr image_data,
           vital::object_track_set_sptr tracks )
{
#ifdef DUMMY_OUTPUT
  typedef vital::track_descriptor td;

  vital::track_descriptor_set_sptr output( new vital::track_descriptor_set() );
  vital::track_descriptor_sptr new_desc = td::create( "cnn_descriptor" );

  td::descriptor_data_sptr_t data( new td::descriptor_data_t( 100 ) );

  for( unsigned i = 0; i < 100; i++ )
  {
    (data->raw_data())[i] = static_cast<double>( i );
  }

  new_desc->set_descriptor( data );

  td::history_entry::image_bbox_t region( 0, 0, image_data->width(), image_data->height() );
  td::history_entry hist_entry( vital::timestamp( 0, 0 ), region );
  new_desc->add_history_entry( hist_entry );

  output->push_back( new_desc );
  return output;
#else
  // Convert inputs to vidtk style inputs
  vil_image_view< vxl_byte > input_image;
  vidtk::timestamp input_ts;
  std::vector<vidtk::track_sptr> input_tracks;

  VITAL_FOREACH( auto vital_t, tracks->tracks() )
  {
    vidtk::track_sptr vidtk_t( new vidtk::track() );
    // TODO: Conversion
  }

  // Run algorithm
  d->m_process.set_source_image( input_image );
  d->m_process.set_source_timestamp( input_ts );
  d->m_process.set_source_tracks( input_tracks );

  if( !image_data )
  {
    d->m_process.flush();
  }
  else if( !d->m_process.step() )
  {

  }

  // Convert outputs to kwiver vital types
  vital::track_descriptor_set_sptr output( new vital::track_descriptor_set() );
  vidtk::raw_descriptor::vector_t computed_desc = d->m_process.descriptors();

  VITAL_FOREACH( auto vidtk_d, computed_desc )
  {
    vital::track_descriptor_sptr vital_d = vital::track_descriptor::create( "test" );
    output->push_back( vital_d );
  }

  return output;
#endif
}


} } } // end namespace
