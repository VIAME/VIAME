// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detect_features_if_keyframe_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/track_features.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( augment_keyframes );

/**
 * \class detect_features_if_keyframe_process
 *
 * \brief detect features if the frame is a keyframe
 *
 * \process This checks if the image is a keyframe in the tracks and
 *          detects features if it is.
 *
 * \iports
 *
 * \iport{timestamp} time stamp for incoming images.
 *
 * \iport{image} Input image to be processed.
 *
 *\iport{next_tracks} Set of detected features from previous image.
 *
 *\iport{loop_back_tracks} Tracks from previous call of
 *                         detect_features_if_keyframe_process
 *
 * \oports
 *
 * \oport{feature_track_set} Set of detected features for input image.
 *
 * \configs
 *
 * \config{track_features} Name of the configuration subblock that selects
 * and configures the feature detector algorithm
 */

//----------------------------------------------------------------
// Private implementation class
class detect_features_if_keyframe_process::priv
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

  // There are many config items for the tracking and stabilization that go
  // directly to the algo.

  algo::track_features_sptr m_tracker;

  const std::string detector_name;

  bool first;

  kwiver::vital::logger_handle_t m_logger;

}; // end priv class

// ================================================================

  detect_features_if_keyframe_process
::detect_features_if_keyframe_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detect_features_if_keyframe_process::priv )
{
  d->m_logger = this->logger();

  make_ports();
  make_config();
}

  detect_features_if_keyframe_process
::~detect_features_if_keyframe_process()
{
}

// ----------------------------------------------------------------
void detect_features_if_keyframe_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Instantiate the configured algorithm
  algo::track_features::set_nested_algo_configuration_using_trait(
    augment_keyframes,
    algo_config,
    d->m_tracker );
  if ( ! d->m_tracker )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Unable to create detect_features_if_keyframe" );
  }

  algo::track_features::get_nested_algo_configuration_using_trait(
    augment_keyframes,
    algo_config,
    d->m_tracker);

  // Check config so it will give run-time diagnostic if any config problems
  // are found
  if ( ! algo::track_features::check_nested_algo_configuration_using_trait(
        augment_keyframes, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Configuration check failed." );
  }
}

// ----------------------------------------------------------------
void
detect_features_if_keyframe_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  //track set including features from the next frame
  kwiver::vital::feature_track_set_sptr next_tracks =
    grab_from_port_as<vital::feature_track_set_sptr>("next_tracks");

  //clone next track set so it can be changed
  next_tracks = std::dynamic_pointer_cast<vital::feature_track_set>(next_tracks->clone());

  //track set from the last call of detect_features_if_keyframe_process::step

  kwiver::vital::feature_track_set_sptr curr_tracks;

  if (!d->first)
  {
    kwiver::vital::feature_track_set_sptr loop_back_tracks =
      grab_from_port_as<vital::feature_track_set_sptr>("loop_back_tracks");

    //merge next_tracks into cur_tracks.  Note, this clones the tracks.
    curr_tracks = d->merge_next_tracks_into_loop_back_track(
      next_tracks, frame_time.get_frame(), loop_back_tracks);
  }
  else
  {
    curr_tracks = next_tracks;
  }
  d->first = false;  //it's not the first call any more

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    // detect features on the current frame
    curr_tracks = d->m_tracker->track(curr_tracks, frame_time.get_frame(), img);
  }

  // return by value
  push_to_port_using_trait(feature_track_set, curr_tracks );
}

// ----------------------------------------------------------------
void detect_features_if_keyframe_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t input_nodep;
  required.insert( flag_required );
  input_nodep.insert(flag_input_nodep );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port("next_tracks", "kwiver:feature_track_set",
    required,
    "tracks including results from later frames");
  declare_input_port("loop_back_tracks", "kwiver:feature_track_set",
    input_nodep,
    "tracks that were output during last call to "
    "detect_features_if_keyframe_process");

  // -- output --
  declare_output_port_using_trait(feature_track_set, required );
}

// ----------------------------------------------------------------
void detect_features_if_keyframe_process
::make_config()
{
  declare_config_using_trait( augment_keyframes );
}

// ================================================================
detect_features_if_keyframe_process::priv
::priv()
  :detector_name("detect_if_keyframe")
  ,first(true)
{
}

detect_features_if_keyframe_process::priv
::~priv()
{
}

vital::feature_track_set_sptr
detect_features_if_keyframe_process::priv
::merge_next_tracks_into_loop_back_track(
  vital::feature_track_set_sptr next_tracks,
  vital::frame_id_t next_tracks_frame_num,
  vital::feature_track_set_sptr loop_back_tracks)
{
  auto next_fd = next_tracks->all_frame_data();
  // clone frame data in place
  for ( auto f : next_fd )
  {
    f.second = f.second->clone();
  }

  vital::feature_track_set_sptr curr_tracks = loop_back_tracks;

  // copy the next frame data into the current tracks.
  curr_tracks->set_frame_data(next_fd);

  // ok, next tracks will have some tracks that are longer or newer than
  // loop_back_tracks.
  std::vector< vital::track_sptr> next_active_tracks =
    next_tracks->active_tracks(next_tracks_frame_num);

  // get the active tracks for the last frame in loop_back tracks.
  vital::frame_id_t last_loop_back_frame_num = loop_back_tracks->last_frame();

  std::vector< vital::track_sptr> curr_active_tracks =
    curr_tracks->active_tracks(last_loop_back_frame_num);

  // Note, track ids from next_tracks and loop_back_tracks do not correspond.
  // KLT tracker never sees detected feature tracks and so it won't increment
  // its IDs to match.

  // need a fast way to search which track a feature is in.  Feature pointers
  // are not cloned, so we can search on those.
  // make a map from feature pointer to curr track pointer.

  typedef std::pair<vital::feature*, vital::track_sptr> feat_track_pair;

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
        feat_track_pair(fts->feature.get() , curr_tk) );
    }
  }

  // what is the next track id in the looped back tracks?
  kwiver::vital::track_id_t next_track_id =
    *curr_tracks->all_track_ids().crbegin() + 1;

  for (auto next_tk : next_active_tracks)
  {
    if (next_tk->size() == 1)
    {
      // This is a new track.  Just add it to curr_tracks
      // Change the track id to account for any new detected features.
      // The KLT tracker was not aware these detected features existed so
      // could not have adjusted its IDs to account for them.
      next_tk->set_id(next_track_id++);

      // clone the track and add it to curr_tracks
      curr_tracks->insert(next_tk->clone());
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
        continue;
      }

      kwiver::vital::feature *feat = fts->feature.get();
      // ok, we have the feature pointer in next active tracks.
      // Let's find it in loop back tracks.
      auto feature_to_curr_track_it = feature_to_curr_track.find(feat);
      if (feature_to_curr_track_it != feature_to_curr_track.end())
      {
        curr_track = feature_to_curr_track_it->second;
        break;
      }
    }
    if (!curr_track)
    {
      // we didn't find the matching track in next_active_tracks
      continue;
    }

    auto ts_clone = next_tk->back()->clone();
    // ok, we have next_tk and curr_track which contain the same features.
    if (!curr_track->append(ts_clone))
    {
      LOG_ERROR(m_logger, "Failed to append track state to loop back track (detect_features_if_keyframe_process)");
    }
    curr_tracks->notify_new_state(ts_clone);
  }

  return curr_tracks;
}

} // end namespace
