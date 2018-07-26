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

#include "close_loops_process.h"

#include <vital/vital_types.h>
#include <vital/io/track_set_io.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/close_loops.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

  create_config_trait( detect_loops, std::string, "",
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
class close_loops_process::priv
{
public:
  priv();
  ~priv();

  vital::feature_track_set_sptr
    merge_next_tracks_into_loop_back_track(
      vital::feature_track_set_sptr next_tracks,
      vital::frame_id_t next_tracks_frame_num,
      vital::feature_track_set_sptr loop_back_tracks);

  // Configuration values

  // There are many config items for the tracking and stabilization that
  // go directly to the algo.

  algo::close_loops_sptr m_loop_closer;

  //static port_t const port_input;
  //static port_type_t const type_custom_feedback;

  bool first;

  kwiver::vital::logger_handle_t m_logger;

  vital::feature_track_set_sptr m_last_output_tracks;
}; // end priv class

// ================================================================

  close_loops_process
::close_loops_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new close_loops_process::priv )
{
  d->m_logger = this->logger();
  make_ports();
  make_config();
}


  close_loops_process
::~close_loops_process()
{
    if (d && d->m_last_output_tracks)
    {
      kwiver::vital::write_feature_track_file(d->m_last_output_tracks,
        "tracks_with_loops.txt");
    }
}


// ----------------------------------------------------------------
void close_loops_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  const std::string algo_name = "close_loops";

  // Instantiate the configured algorithm
  algo::close_loops::set_nested_algo_configuration(algo_name, algo_config,
    d->m_loop_closer );

  if ( ! d->m_loop_closer )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Unable to create close_loops" );
  }

  algo::close_loops::get_nested_algo_configuration(algo_name, algo_config,
    d->m_loop_closer);

  //// Check config so it will give run-time diagnostic if any config problems
  // are found
  if ( ! algo::close_loops::check_nested_algo_configuration(
      algo_name, algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
close_loops_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  vital::feature_track_set_sptr next_tracks =
    grab_from_port_as<vital::feature_track_set_sptr>("next_tracks");

  vital::feature_track_set_sptr curr_tracks;
  if (!d->first)
  {
    vital::feature_track_set_sptr loob_back_tracks =
      grab_from_port_as<vital::feature_track_set_sptr>("loop_back_tracks");

    curr_tracks = d->merge_next_tracks_into_loop_back_track(
                   next_tracks, frame_time.get_frame(), loob_back_tracks);
  }
  else
  {
    curr_tracks =
      std::dynamic_pointer_cast<vital::feature_track_set>(next_tracks->clone());
  }
  d->first = false;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "detecting loops with frame " << frame_time );

    // detect features on the current frame
    curr_tracks = d->m_loop_closer->stitch(frame_time.get_frame(),curr_tracks,kwiver::vital::image_container_sptr());
  }

  // return by value
  push_to_port_using_trait(feature_track_set, curr_tracks );
  d->m_last_output_tracks = curr_tracks;
}


// ----------------------------------------------------------------
void close_loops_process
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

  declare_input_port("next_tracks", "kwiver:feature_track_set", required,
    "feature track set for the next frame.  Features are not yet matched for "
    "any frames.");

  declare_input_port("loop_back_tracks", "kwiver:feature_track_set", input_nodep,
    "feature track set from last call to detect loops.  May include loops "
    "(joined track sets, split track sets.");

  // -- output --
  declare_output_port_using_trait(feature_track_set, optional );
}


// ----------------------------------------------------------------
void close_loops_process
::make_config()
{
  declare_config_using_trait( detect_loops );
}


// ================================================================
close_loops_process::priv
::priv():first(true)
{
}


close_loops_process::priv
::~priv()
{
}

vital::feature_track_set_sptr
close_loops_process::priv
::merge_next_tracks_into_loop_back_track(
  vital::feature_track_set_sptr next_tracks,
  vital::frame_id_t next_tracks_frame_num,
  vital::feature_track_set_sptr loop_back_tracks)
{
  //clone loop back tracks so we can change it.
  vital::feature_track_set_sptr curr_tracks =
    std::dynamic_pointer_cast<vital::feature_track_set>(loop_back_tracks->clone());

  //need to pull the key-frame data from next_tracks
  curr_tracks->set_frame_data(next_tracks->all_frame_data());

  // ok, next tracks will have some tracks that are longer or newer than
  // loop_back_tracks.
  std::vector< vital::track_sptr> next_active_tracks =
    next_tracks->active_tracks(next_tracks_frame_num);
  //get the active tracks for the last frame in loop_back tracks.
  std::vector< vital::track_sptr> curr_active_tracks =
    curr_tracks->active_tracks(next_tracks_frame_num - 1);

  // Note, track ids from next_tracks and loop_back_tracks do not correspond.
  // KLT tracker never sees detected feature tracks and so it won't increment
  // its IDs to match.

  // need a fast way to search which track a feature is in.  Feature pointers
  // are not cloned, so we can search on those. Make a map from feature pointer
  // to loop back track pointer.

  std::map<vital::feature *, vital::track_sptr> feature_to_curr_track;
  for (auto curr_tk : curr_active_tracks)
  {
    for (auto curr_tk_state = curr_tk->begin();
           curr_tk_state != curr_tk->end(); ++curr_tk_state)
    {
      vital::feature_track_state_sptr fts =
        std::dynamic_pointer_cast<vital::feature_track_state>(*curr_tk_state);
      if (!fts)
      {
        continue;
      }
      feature_to_curr_track.insert(
        std::pair<vital::feature*,
                  vital::track_sptr>(fts->feature.get(), curr_tk));
    }
  }

  for (auto next_tk : next_active_tracks)
  {
    if (next_tk->size() == 1)
    {
      //this is a new track.  Just add it to curr_tracks
      curr_tracks->insert(next_tk->clone()); // clone the track and add it to
                                             // curr_tracks
      continue;
    }

    vital::track_sptr curr_track;
    for (auto next_tk_state = next_tk->begin();
      next_tk_state != next_tk->end(); ++next_tk_state)
    {
      vital::feature_track_state_sptr fts =
        std::dynamic_pointer_cast<vital::feature_track_state>(*next_tk_state);
      if (!fts)
      {
        LOG_ERROR(m_logger,
          "fts should have cast to a feature_track_state and didn't");
        continue;
      }

      kwiver::vital::feature *feat = fts->feature.get();
      // ok, we have the feature pointer in next active tracks.  Let's find it
      // in loop back tracks.
      auto feature_to_curr_track_it = feature_to_curr_track.find(feat);
      if (feature_to_curr_track_it != feature_to_curr_track.end())
      {
        curr_track = feature_to_curr_track_it->second;
        break;
      }
    }
    if (!curr_track)
    {
      //we didn't find the matching track in cur_tracks
      LOG_ERROR(m_logger,
        "We should have found a matching track in cur_tracks and didn't");
      continue;
    }

    auto ts_clone = next_tk->back()->clone();
    //ok, we have next_tk and curr_track which contain the same features.
    if (!curr_track->append(ts_clone))
    {
      LOG_ERROR(m_logger, "Failed to append track state to loop back track (close_loops_process");
    }
    curr_tracks->notify_new_state(ts_clone);
  }

  return curr_tracks;
}

} // end namespace
