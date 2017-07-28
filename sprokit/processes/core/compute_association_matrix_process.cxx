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

#include "compute_association_matrix_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/types/matrix.h>

#include <vital/algo/compute_association_matrix.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace kwiver
{

namespace algo = vital::algo;

//------------------------------------------------------------------------------
// Private implementation class
class compute_association_matrix_process::priv
{
public:
  priv();
  ~priv();

  unsigned track_read_delay;

  algo::compute_association_matrix_sptr m_matrix_generator;
}; // end priv class


// =============================================================================

compute_association_matrix_process
::compute_association_matrix_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new compute_association_matrix_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  // Required for negative feedback loop
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


compute_association_matrix_process
::~compute_association_matrix_process()
{
}


// -----------------------------------------------------------------------------
void compute_association_matrix_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  algo::compute_association_matrix::set_nested_algo_configuration(
    "matrix_generator", algo_config, d->m_matrix_generator );

  if( !d->m_matrix_generator )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create compute_association_matrix" );
  }

  algo::compute_association_matrix::get_nested_algo_configuration(
    "matrix_generator", algo_config, d->m_matrix_generator );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::compute_association_matrix::check_nested_algo_configuration(
    "matrix_generator", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }
}


// -----------------------------------------------------------------------------
void
compute_association_matrix_process
::_step()
{
  // Check for end of video since we're in manual management mode
  auto port_info = peek_at_port_using_trait( detected_object_set );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( detected_object_set );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( matrix_d, dat );
    push_datum_to_port_using_trait( object_track_set, dat );
    push_datum_to_port_using_trait( detected_object_set, dat );
    return;
  }

  // Retrieve inputs from ports
  vital::timestamp frame_id;
  vital::image_container_sptr image;
  vital::object_track_set_sptr tracks;
  vital::detected_object_set_sptr detections;

  detections = grab_from_port_using_trait( detected_object_set );

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

  if( d->track_read_delay > 0 )
  {
    d->track_read_delay--;

    tracks = vital::object_track_set_sptr( new vital::simple_object_track_set() );
  }
  else
  {
    tracks = grab_from_port_using_trait( object_track_set );
  }

  // Get output matrix and detections
  vital::matrix_d matrix_output;
  vital::detected_object_set_sptr detection_output;

  d->m_matrix_generator->compute( frame_id, image, tracks,
    detections, matrix_output, detection_output );

  // Return all outputs
  push_to_port_using_trait( matrix_d, matrix_output );
  push_to_port_using_trait( object_track_set, tracks );
  push_to_port_using_trait( detected_object_set, detection_output );
}


// -----------------------------------------------------------------------------
void compute_association_matrix_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t required_no_dep;

  required.insert( flag_required );
  required_no_dep.insert( flag_required );
  required_no_dep.insert( flag_input_nodep );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( object_track_set, required_no_dep );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( matrix_d, optional );
  declare_output_port_using_trait( object_track_set, optional );
  declare_output_port_using_trait( detected_object_set, optional );
}


// -----------------------------------------------------------------------------
void compute_association_matrix_process
::make_config()
{
}


// =============================================================================
compute_association_matrix_process::priv
::priv()
  : track_read_delay( 1 )
{
}


compute_association_matrix_process::priv
::~priv()
{
}

} // end namespace
