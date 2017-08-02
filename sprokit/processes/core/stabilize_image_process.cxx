/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include "stabilize_image_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/homography.h>

#include <vital/algo/track_features.h>
#include <vital/algo/compute_ref_homography.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class stabilize_image_process::priv
{
public:
  priv();
  ~priv();


  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  // feature tracker algorithm - homography source
  algo::track_features_sptr         m_feature_tracker;
  algo::compute_ref_homography_sptr m_compute_homog;

  vital::feature_track_set_sptr m_tracks; // last set of tracks

}; // end priv class

// ================================================================

stabilize_image_process
::stabilize_image_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new stabilize_image_process::priv )
{
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


stabilize_image_process
::~stabilize_image_process()
{
}


// ----------------------------------------------------------------
void stabilize_image_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::track_features::set_nested_algo_configuration( "track_features", algo_config, d->m_feature_tracker );
  if ( ! d->m_feature_tracker )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create \"track_features\"" );
  }
  algo::track_features::get_nested_algo_configuration( "track_features", algo_config, d->m_feature_tracker );

  // ----
  algo::compute_ref_homography::set_nested_algo_configuration( "homography_generator", algo_config, d->m_compute_homog );
  if ( ! d->m_compute_homog )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create \"compute_ref_homography\"" );
  }
  algo::compute_ref_homography::get_nested_algo_configuration( "homography_generator", algo_config, d->m_compute_homog );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! algo::track_features::check_nested_algo_configuration( "track_features", algo_config ) ||
       ! algo::compute_ref_homography::check_nested_algo_configuration("homography_generator", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
stabilize_image_process
::_step()
{
  kwiver::vital::f2f_homography_sptr src_to_ref_homography;

  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  // LOG_DEBUG - this is a good thing to have in all processes that handle frames.
  LOG_DEBUG( logger(), "Processing frame " << frame_time );

  // Get feature tracks
  d->m_tracks = d->m_feature_tracker->track( d->m_tracks,
                                             static_cast<unsigned int>(frame_time.get_frame()),
                                             img );

  // Get stabilization homography
  src_to_ref_homography = d->m_compute_homog->estimate( frame_time.get_frame(), d->m_tracks );

  // return by value
  push_to_port_using_trait( homography_src_to_ref, *src_to_ref_homography );
}


// ----------------------------------------------------------------
void stabilize_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );

  declare_output_port_using_trait( homography_src_to_ref, optional );
}


// ----------------------------------------------------------------
void stabilize_image_process
::make_config()
{

}


// ================================================================
stabilize_image_process::priv
::priv()
{
}


stabilize_image_process::priv
::~priv()
{
}

} // end namespace
