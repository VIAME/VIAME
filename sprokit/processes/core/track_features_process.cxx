/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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

#include "track_features_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/track_features.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

  create_config_trait( track_features, std::string, "",
    "Algorithm configuration subblock." )

/**
 * \class track_features_process
 *
 * \brief track feature points in supplied images.
 *
 * \process This process generates a list of tracked features that
 * can be used to determine coordinate transforms between images. The
 * actual tracking is done by the selected \b track_features
 * algorithm implementation
 *
 * \iports
 *
 * \iport{timestamp} time stamp for incoming images.
 *
 * \iport{image} Input image to be processed.
 *
 *\iport{feature_set} Set of detected features from previous image.
 *
 * \oports
 *
 * \oport{feature_set} Set of detected features for input image.
 *
 * \configs
 *
 * \config{track_features} Name of the configuration subblock that selects
 * and configures the feature detector algorithm
 */

//----------------------------------------------------------------
// Private implementation class
class track_features_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values

  // There are many config items for the tracking and stabilization that go
  // directly to the algo.

  algo::track_features_sptr m_tracker;

  //static port_t const port_input;
  //static port_type_t const type_custom_feedback;

  bool first;

}; // end priv class

// ================================================================

 track_features_process
::track_features_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new track_features_process::priv )
{
  make_ports();
  make_config();
}


 track_features_process
::~track_features_process()
{
}


// ----------------------------------------------------------------
void track_features_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Instantiate the configured algorithm
  algo::track_features::set_nested_algo_configuration( "track_features",
    algo_config, d->m_tracker );

  if ( ! d->m_tracker )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Unable to create track_features" );
  }

  algo::track_features::get_nested_algo_configuration("track_features",
    algo_config, d->m_tracker);

  //// Check config so it will give run-time diagnostic if any config problems
  // are found
  if ( ! algo::track_features::check_nested_algo_configuration(
          "track_features", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
track_features_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );


  kwiver::vital::feature_track_set_sptr cur_tracks;
  if (!d->first)
  {
    kwiver::vital::feature_track_set_sptr prev_tracks =
      grab_from_port_using_trait(feature_track_set);
    // clone prev tracks.  This way any changes made on it by m_tracker are done
    // to a unique object.
    cur_tracks =
      std::dynamic_pointer_cast<vital::feature_track_set>(prev_tracks->clone());
  }
  d->first = false;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    // detect features on the current frame
    cur_tracks = d->m_tracker->track(cur_tracks, frame_time.get_frame(), img);
  }

  // return by value
  push_to_port_using_trait(feature_track_set, cur_tracks);
}


// ----------------------------------------------------------------
void track_features_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t input_nodep;
  required.insert( flag_required );
  input_nodep.insert(flag_input_nodep );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait(feature_track_set, input_nodep);

  // -- output --
  declare_output_port_using_trait(feature_track_set, optional );
}


// ----------------------------------------------------------------
void track_features_process
::make_config()
{
  declare_config_using_trait( track_features );
}


// ================================================================
track_features_process::priv
::priv():first(true)
{
}


track_features_process::priv
::~priv()
{
}

} // end namespace
