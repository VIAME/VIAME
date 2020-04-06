/*ckwg +29
 * Copyright 2015-2017, 2020 by Kitware, Inc.
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

#include <arrows/core/track_set_impl.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( keyframe_selection_1 );

create_port_trait( next_tracks, feature_track_set,
                   "feature track set for the next frame.");

create_port_trait( loop_back_tracks, feature_track_set,
                   "feature track set from last call to keyframe_selection_process.");

create_port_trait( only_frame_data_tracks, feature_track_set,
                   "output track set with only frame data, no klt tracks.");

create_port_trait( to_loop_back_tracks, feature_track_set,
                   "accumulated klt tracks");


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

  vital::feature_track_set_sptr
  merge_next_tracks_into_loop_back_track(
    vital::feature_track_set_sptr next_tracks,
    vital::frame_id_t next_tracks_frame_num,
    vital::feature_track_set_sptr loop_back_tracks);

  vital::feature_track_set_sptr
  remove_non_keyframes_between_keyframes(
      vital::feature_track_set_sptr input_tracks,
      vital::frame_id_t current_frame_number);

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  algo::keyframe_selection_sptr m_keyframe_selection;
  bool first_frame;
  vital::frame_id_t m_earliest_checked_frame_id;

}; // end priv class

// ================================================================

 keyframe_selection_process
::keyframe_selection_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new keyframe_selection_process::priv )
{
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

  // Instantiate the configured algorithm
  algo::keyframe_selection::set_nested_algo_configuration_using_trait(
    keyframe_selection_1,
    algo_config,
    d->m_keyframe_selection );
  if ( ! d->m_keyframe_selection )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create keyframe_selection" );
  }

  algo::keyframe_selection::get_nested_algo_configuration_using_trait(
    keyframe_selection_1,
    algo_config,
    d->m_keyframe_selection);

  //// Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::keyframe_selection::check_nested_algo_configuration_using_trait(
         keyframe_selection_1, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------
void
keyframe_selection_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );
  vital::feature_track_set_sptr next_tracks = grab_from_port_using_trait( next_tracks );

  next_tracks = std::dynamic_pointer_cast<vital::feature_track_set>(next_tracks->clone());

  vital::feature_track_set_sptr curr_tracks;
  if (!d->first_frame)
  {
    vital::feature_track_set_sptr loop_back_tracks = grab_from_port_using_trait( loop_back_tracks );

    loop_back_tracks = std::dynamic_pointer_cast<vital::feature_track_set>(loop_back_tracks->clone());

    //merging does the clone of next tracks
    curr_tracks = d->merge_next_tracks_into_loop_back_track(
      next_tracks, frame_time.get_frame(), loop_back_tracks);
  }
  else
  {
    curr_tracks = next_tracks;
  }


  d->first_frame = false;

  kwiver::vital::feature_track_set_sptr new_kf_tracks;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Selecting keyframes " << frame_time );

    // detect features on the current frame
    kwiver::vital::track_set_sptr new_kf_tracks_track_set;
    new_kf_tracks_track_set = d->m_keyframe_selection->select(curr_tracks);
    new_kf_tracks = std::dynamic_pointer_cast<kwiver::vital::feature_track_set>(new_kf_tracks_track_set);

    new_kf_tracks = d->remove_non_keyframes_between_keyframes(new_kf_tracks, frame_time.get_frame());

  }

  // return by value
  push_to_port_using_trait(to_loop_back_tracks, new_kf_tracks);

  typedef std::unique_ptr<vital::track_set_implementation> tsi_uptr;

  vital::feature_track_set_sptr continuing_tracks = std::make_shared<vital::feature_track_set>(
    tsi_uptr(new kwiver::arrows::core::frame_index_track_set_impl()));

  continuing_tracks->set_frame_data(new_kf_tracks->all_frame_data());

  push_to_port_using_trait(only_frame_data_tracks, continuing_tracks);
}


// ----------------------------------------------------------------
void keyframe_selection_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t no_dep;
  required.insert( flag_required );
  no_dep.insert( flag_input_nodep );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( next_tracks, required );
  declare_input_port_using_trait( loop_back_tracks, no_dep );

  // -- output --
  declare_input_port_using_trait( only_frame_data_tracks, required );
  declare_input_port_using_trait( to_loop_back_tracks, required );
}


// ----------------------------------------------------------------
void keyframe_selection_process
::make_config()
{
  declare_config_using_trait( keyframe_selection_1 );
}


// ================================================================
keyframe_selection_process::priv
::priv()
  :first_frame(true),
   m_earliest_checked_frame_id(-1)
{
}

//----------------------------------------------------------------

keyframe_selection_process::priv
::~priv()
{
}

//----------------------------------------------------------------

vital::feature_track_set_sptr
keyframe_selection_process::priv
::remove_non_keyframes_between_keyframes(
  vital::feature_track_set_sptr input_tracks,
  vital::frame_id_t current_frame_number)
{
  bool passed_a_keyframe = false;
  auto afd = input_tracks->all_frame_data();
  std::vector<vital::frame_id_t> removed_frames;

  vital::frame_id_t latest_keyframe = m_earliest_checked_frame_id;

  for (auto it = afd.rbegin(); it != afd.rend(); ++it)
  {
    vital::frame_id_t fn = it->first;
    if (fn <= m_earliest_checked_frame_id)
    {
      //no need to check before this frame
      break;
    }
    auto fd_cur =
      std::dynamic_pointer_cast<
         kwiver::vital::feature_track_set_frame_data>(it->second);
    if (fd_cur && fd_cur->is_keyframe)
    {
      if (fn > latest_keyframe)
      {
        latest_keyframe = fn;
      }
      passed_a_keyframe = true;
    }
    else
    {
      if (passed_a_keyframe)
      {
        // we have passed a keyframe going backwards in time and the current
        // frame is not a keyframe.  So we remove this one.
        auto fn_states = input_tracks->frame_states(fn);
        for (auto &fn_state : fn_states)
        {
          auto t = fn_state->track();
          t->remove(fn_state);
          input_tracks->notify_removed_state(fn_state);
          if (t->empty())
          {
            input_tracks->remove(t);
          }
        }
        removed_frames.push_back(fn);
      }
    }
  }
  for (auto f : removed_frames)
  {
    input_tracks->remove_frame_data(f);
  }

  m_earliest_checked_frame_id = latest_keyframe;

  return input_tracks;

}

//----------------------------------------------------------------

vital::feature_track_set_sptr
keyframe_selection_process::priv
::merge_next_tracks_into_loop_back_track(
  vital::feature_track_set_sptr next_tracks,
  vital::frame_id_t next_tracks_frame_num,
  vital::feature_track_set_sptr loop_back_tracks)
{
  vital::feature_track_set_sptr curr_tracks = loop_back_tracks;

  curr_tracks->merge_in_other_track_set(next_tracks);

  return curr_tracks;
}

} // end namespace
