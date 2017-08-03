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

#include "compute_track_descriptors_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/types/track_descriptor_set.h>

#include <vital/algo/compute_track_descriptors.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

namespace algo = vital::algo;

//------------------------------------------------------------------------------
// Private implementation class
class compute_track_descriptors_process::priv
{
public:
  priv();
  ~priv();

  unsigned track_read_delay;

  algo::compute_track_descriptors_sptr m_computer;
}; // end priv class


// =============================================================================

compute_track_descriptors_process
::compute_track_descriptors_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new compute_track_descriptors_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}


compute_track_descriptors_process
::~compute_track_descriptors_process()
{
}


// -----------------------------------------------------------------------------
void compute_track_descriptors_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  algo::compute_track_descriptors::set_nested_algo_configuration(
    "computer", algo_config, d->m_computer );

  if( !d->m_computer )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create compute_track_descriptors" );
  }

  algo::compute_track_descriptors::get_nested_algo_configuration(
    "computer", algo_config, d->m_computer );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::compute_track_descriptors::check_nested_algo_configuration(
    "computer", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }
}


// -----------------------------------------------------------------------------
void
compute_track_descriptors_process
::_step()
{
  // Retrieve inputs from ports
  vital::timestamp frame_id;
  vital::image_container_sptr image;
  vital::object_track_set_sptr tracks;
  vital::detected_object_set_sptr detections;

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

  if( process::has_input_port_edge( "detected_object_set" ) )
  {
    detections = grab_from_port_using_trait( detected_object_set );
  }

  if( process::has_input_port_edge( "object_track_set" ) )
  {
    tracks = grab_from_port_using_trait( object_track_set );
  }

  // Process optional input track set
  vital::track_descriptor_set_sptr output;

  if( process::has_input_port_edge( "object_track_set" ) )
  {
    output = d->m_computer->compute( image, tracks );
  }

  // Process optional input detection set
  // [TODO]

  // Return all outputs
  push_to_port_using_trait( track_descriptor_set, output );
}


// -----------------------------------------------------------------------------
void compute_track_descriptors_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( object_track_set, optional );
  declare_input_port_using_trait( detected_object_set, optional );

  // -- output --
  declare_output_port_using_trait( track_descriptor_set, optional );
}


// -----------------------------------------------------------------------------
void compute_track_descriptors_process
::make_config()
{
}


// =============================================================================
compute_track_descriptors_process::priv
::priv()
{
}


compute_track_descriptors_process::priv
::~priv()
{
}

} // end namespace
