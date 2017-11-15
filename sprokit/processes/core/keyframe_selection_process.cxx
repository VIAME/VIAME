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

#include "keyframe_selection_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/keyframe_selection.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

  create_config_trait( keyframe_selection, std::string, "", "Algorithm configuration subblock." )

/**
 * \class keyframe_selection_process
 *
 * \brief select keyframes based on the supplied tracks.
 *
 * \process This process generates a list of tracked features that
 * can be used to determine coordinate transforms between images. The
 * actual tracking is done by the selected \b track_features
 * algorithm implementation
 *
 * \iports
 *\iport{timestamp} Time stamp when keyframe selection was requested
 *
 *\iport{feature_set} Feature track set, may include previous keyframe selections
 *
 * \oports
 *
 * \oport{feature_set} Feature track set with updated keyframe selections
 *
 * \configs
 *
 * \config{keyframe_selection} Name of the configuration subblock that selects
 * and configures the feature detector algorithm
 */

//----------------------------------------------------------------
// Private implementation class
class keyframe_selection_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  algo::keyframe_selection_sptr m_keyframe_selection;

}; // end priv class

// ================================================================

 keyframe_selection_process
::keyframe_selection_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new keyframe_selection_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


 keyframe_selection_process
::~keyframe_selection_process()
{
}


// ----------------------------------------------------------------
void keyframe_selection_process
::_configure()
{
  scoped_configure_instrumentation();


  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  const std::string algo_name = "keyframe_selection_1";

  // Instantiate the configured algorithm
  algo::keyframe_selection::set_nested_algo_configuration(algo_name, algo_config, d->m_keyframe_selection );
  if ( ! d->m_keyframe_selection )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create keyframe_selection" );
  }

  algo::keyframe_selection::get_nested_algo_configuration(algo_name, algo_config, d->m_keyframe_selection);

  //// Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::keyframe_selection::check_nested_algo_configuration(algo_name, algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------
void
keyframe_selection_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  kwiver::vital::feature_track_set_sptr prev_tracks = grab_from_port_using_trait(feature_track_set);

  kwiver::vital::feature_track_set_sptr new_kf_tracks;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Selecting keyframes " << frame_time );

    kwiver::vital::track_set_sptr prev_tracks_track_set = std::dynamic_pointer_cast<kwiver::vital::track_set>(prev_tracks);

    // detect features on the current frame
    kwiver::vital::track_set_sptr new_kf_tracks_track_set;
    new_kf_tracks_track_set = d->m_keyframe_selection->select(prev_tracks);
    new_kf_tracks = std::dynamic_pointer_cast<kwiver::vital::feature_track_set>(new_kf_tracks_track_set);
  }

  // return by value
  push_to_port_using_trait(feature_track_set, new_kf_tracks );
}


// ----------------------------------------------------------------
void keyframe_selection_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( feature_track_set, required);

  // -- output --
  declare_output_port_using_trait(feature_track_set, required );
}


// ----------------------------------------------------------------
void keyframe_selection_process
::make_config()
{
  declare_config_using_trait( keyframe_selection );
}


// ================================================================
keyframe_selection_process::priv
::priv()
{
}


keyframe_selection_process::priv
::~priv()
{
}

} // end namespace
