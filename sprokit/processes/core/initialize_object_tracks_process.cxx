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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "initialize_object_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>

#include <vital/algo/initialize_object_tracks.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

namespace algo = vital::algo;

//------------------------------------------------------------------------------
// Private implementation class
class initialize_object_tracks_process::priv
{
public:
  priv();
  ~priv();

  algo::initialize_object_tracks_sptr m_track_initializer;
}; // end priv class


// =============================================================================

initialize_object_tracks_process
::initialize_object_tracks_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new initialize_object_tracks_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}


initialize_object_tracks_process
::~initialize_object_tracks_process()
{
}


// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  algo::initialize_object_tracks::set_nested_algo_configuration(
    "track_initializer", algo_config, d->m_track_initializer );

  if( !d->m_track_initializer )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create initialize_object_tracks" );
  }

  algo::initialize_object_tracks::get_nested_algo_configuration(
    "track_initializer", algo_config, d->m_track_initializer );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::initialize_object_tracks::check_nested_algo_configuration(
    "track_initializer", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }
}


// -----------------------------------------------------------------------------
void
initialize_object_tracks_process
::_step()
{
  vital::timestamp frame_id;
  vital::image_container_sptr image;
  vital::detected_object_set_sptr detections;
  vital::object_track_set tracks;

  vital::object_track_set output;

  frame_id = grab_from_port_using_trait( timestamp );
  image = grab_from_port_using_trait( image );
  detections = grab_from_port_using_trait( detected_object_set );
  tracks = grab_from_port_using_trait( object_track_set );

  // Output frame ID
  LOG_DEBUG( logger(), "Processing frame " << frame_id );

  // Get stabilization homography
  output = d->m_track_initializer->initialize( frame_id, image, detections );

  // Union optional input tracks if available
  // [TODO]

  // Return by value
  push_to_port_using_trait( object_track_set, output );
}


// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}


// -----------------------------------------------------------------------------
void initialize_object_tracks_process
::make_config()
{

}


// =============================================================================
initialize_object_tracks_process::priv
::priv()
{
}


initialize_object_tracks_process::priv
::~priv()
{
}

} // end namespace
