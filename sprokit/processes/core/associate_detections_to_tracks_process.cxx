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

#include "associate_detections_to_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/types/matrix.h>

#include <vital/algo/associate_detections_to_tracks.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

namespace algo = vital::algo;

create_port_trait( unused_detections,
  detected_object_set,
  "Set of detected objects not linked to any tracks." );

create_port_trait( all_detections,
  detected_object_set,
  "Set of all detected objects for the given frame." );

//------------------------------------------------------------------------------
// Private implementation class
class associate_detections_to_tracks_process::priv
{
public:
  priv();
  ~priv();

  algo::associate_detections_to_tracks_sptr m_track_associator;
}; // end priv class


// =============================================================================

associate_detections_to_tracks_process
::associate_detections_to_tracks_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new associate_detections_to_tracks_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}


associate_detections_to_tracks_process
::~associate_detections_to_tracks_process()
{
}


// -----------------------------------------------------------------------------
void associate_detections_to_tracks_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  algo::associate_detections_to_tracks::set_nested_algo_configuration(
    "track_associator", algo_config, d->m_track_associator );

  if( !d->m_track_associator )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create associate_detections_to_tracks" );
  }

  algo::associate_detections_to_tracks::get_nested_algo_configuration(
    "track_associator", algo_config, d->m_track_associator );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::associate_detections_to_tracks::check_nested_algo_configuration(
    "track_associator", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }
}


// -----------------------------------------------------------------------------
void
associate_detections_to_tracks_process
::_step()
{
  vital::timestamp frame_id;
  vital::image_container_sptr image;
  vital::object_track_set_sptr tracks;
  vital::detected_object_set_sptr detections;
  vital::matrix_d ass_matrix;

  if( process::has_input_port_edge( "timestamp" ) )
  {
    frame_id = grab_from_port_using_trait( timestamp );

    // Output frame ID
    LOG_DEBUG( logger(), "Processing frame " << frame_id );
  }

  if( process::has_input_port_edge( "image" ) )
  {
    image = grab_from_port_using_trait( image );
  }

  tracks = grab_from_port_using_trait( object_track_set );
  detections = grab_from_port_using_trait( detected_object_set );
  ass_matrix = grab_from_port_using_trait( matrix_d );

  vital::object_track_set_sptr output;
  vital::detected_object_set_sptr unused;

  // Run associator
  d->m_track_associator->associate( frame_id, image,
    tracks, detections, ass_matrix, output, unused );

  // Return by value
  push_to_port_using_trait( object_track_set, output );
  push_to_port_using_trait( all_detections, detections );
  push_to_port_using_trait( unused_detections, unused );
}


// -----------------------------------------------------------------------------
void associate_detections_to_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( object_track_set, required );
  declare_input_port_using_trait( detected_object_set, required );
  declare_input_port_using_trait( matrix_d, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
  declare_output_port_using_trait( all_detections, optional );
  declare_output_port_using_trait( unused_detections, optional );
}


// -----------------------------------------------------------------------------
void associate_detections_to_tracks_process
::make_config()
{

}


// =============================================================================
associate_detections_to_tracks_process::priv
::priv()
{
}


associate_detections_to_tracks_process::priv
::~priv()
{
}

} // end namespace
