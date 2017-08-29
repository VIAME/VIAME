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

#include <arrows/vxl/image_container.h>

#include <vital/exceptions.h>
#include <vital/vital_foreach.h>

#include <descriptors/online_descriptor_computer_process.h>


namespace kwiver {
namespace arrows {
namespace burnout {


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

  config->set_value( "config_file", d->m_config_file,  "Name of config file." );

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
  d->m_config_file = config->get_value< std::string >( "config_file" );

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
  std::string config_fn = config->get_value< std::string >( "config_file" );

  if( config_fn.empty() )
  {
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------------
vidtk::timestamp
vital_to_vidtk( vital::timestamp ts )
{
  return vidtk::timestamp( ts.get_frame(), ts.get_time_usec() );
}

vidtk::timestamp
vital_to_vidtk( unsigned fid )
{
  vidtk::timestamp output;
  output.set_frame_number( fid );
  return output;
}

vidtk::track_state_sptr
vital_to_vidtk( const vital::object_track_state* ots )
{
  vidtk::track_state_sptr output( new vidtk::track_state() );
  output->set_timestamp( vital_to_vidtk( ots->frame() ) );

  if( ots->detection )
  {
    auto bbox = ots->detection->bounding_box();
    vidtk::image_object_sptr iobj( new vidtk::image_object() );
    iobj->set_bbox( bbox.min_x(), bbox.max_x(), bbox.min_y(), bbox.max_y() );
    output->set_image_object( iobj );
  }

  return output;
}

vital::bounding_box_d
vidtk_to_vital( vgl_box_2d< unsigned > box )
{
  return vital::bounding_box_d(
    box.min_x(), box.min_y(), box.max_x(), box.max_y() );
}

vital::bounding_box_d
vidtk_to_vital( vgl_box_2d< double > box )
{
  return vital::bounding_box_d(
    box.min_x(), box.min_y(), box.max_x(), box.max_y() );
}

vital::timestamp
vidtk_to_vital( vidtk::timestamp ts )
{
  return vital::timestamp( ts.frame_number(), ts.time() );
}


// ----------------------------------------------------------------------------------
vital::track_descriptor_set_sptr
burnout_track_descriptors
::compute( vital::timestamp ts,
           vital::image_container_sptr image_data,
           vital::object_track_set_sptr tracks )
{
#ifndef DUMMY_OUTPUT

  // Convert inputs to burnout style inputs
  vidtk::timestamp input_ts = vital_to_vidtk( ts );
  vil_image_view< vxl_byte > input_image;
  std::vector< vidtk::track_sptr > input_tracks;

  if( tracks )
  {
    VITAL_FOREACH( auto vital_t, tracks->tracks() )
    {
      vidtk::track_sptr vidtk_t( new vidtk::track() );

      vidtk_t->set_id( vital_t->id() );

      VITAL_FOREACH( auto vital_ts, *vital_t )
      {
        vital::object_track_state* ots =
          dynamic_cast< vital::object_track_state* >( vital_ts.get() );

        if( ots )
        {
          vidtk_t->add_state( vital_to_vidtk( ots ) );
        }
      }

      input_tracks.push_back( vidtk_t );
    }
  }

  if( image_data )
  {
    input_image = vxl::image_container::vital_to_vxl( image_data->get_image() );
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
    throw std::runtime_error( "Unable to step burnout descriptor process" );
  }

  // Convert outputs to kwiver vital types
  vital::track_descriptor_set_sptr output( new vital::track_descriptor_set() );
  vidtk::raw_descriptor::vector_t computed_desc = d->m_process.descriptors();

  VITAL_FOREACH( auto vidtk_d, computed_desc )
  {
    vital::track_descriptor_sptr vital_d =
      vital::track_descriptor::create( vidtk_d->get_type() );

    auto vidtk_rd = vidtk_d->get_features();

    if( vidtk_rd.empty() )
    {
      continue;
    }

    vital::track_descriptor::descriptor_data_sptr vital_rd(
      new vital::track_descriptor::descriptor_data_t(
        vidtk_rd.size(), &vidtk_rd[0] ) );

    vital_d->set_descriptor( vital_rd );

    VITAL_FOREACH( auto id, vidtk_d->get_track_ids() )
    {
      vital_d->add_track_id( id );
    }

    VITAL_FOREACH( auto hist_ent, vidtk_d->get_history() )
    {
      vital::track_descriptor::history_entry vital_ent(
        vidtk_to_vital( hist_ent.get_timestamp() ),
        vidtk_to_vital( hist_ent.get_image_location() ),
        vidtk_to_vital( hist_ent.get_world_location() ) );

      vital_d->add_history_entry( vital_ent );
    }

    output->push_back( vital_d );
  }

  return output;
#else
  // Generate simple dummy example output.
  // This function is used for testing GUIs, amongst other things.

  typedef vital::track_descriptor td;

  vital::track_descriptor_set_sptr output( new vital::track_descriptor_set() );
  vital::track_descriptor_sptr new_desc = td::create( "cnn_descriptor" );

  td::descriptor_data_sptr data( new td::descriptor_data_t( 100 ) );

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
#endif
}


vital::track_descriptor_set_sptr
burnout_track_descriptors
::flush()
{
  return compute( vital::timestamp(),
    vital::image_container_sptr(),
    vital::object_track_set_sptr() );
}

} } } // end namespace
