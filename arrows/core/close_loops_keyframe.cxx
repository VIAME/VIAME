/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief Implementation of close_loops_keyframe
 */

#include "close_loops_keyframe.h"

#include <functional>
#include <future>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <vital/exceptions/algorithm.h>
#include <vital/algo/match_features.h>


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;

namespace {
/// Functor to help remove tracks from vector
bool track_in_set( track_sptr trk_ptr, std::set<track_id_t>* set_ptr )
{
  return set_ptr->find( trk_ptr->id() ) != set_ptr->end();
}


typedef std::vector<std::pair<track_sptr, track_sptr> > track_pairs;

/// A helper funtion to identify matching tracks across frames
class track_matcher
{
public:
  /// Constructor
  track_matcher(vital::algo::match_features_sptr m,
                track_set_sptr at,
                std::vector<track_sptr> const& ct,
                feature_set_sptr cf,
                descriptor_set_sptr cd)
    : matcher(m)
    , all_tracks(at)
    , current_tracks(ct)
    , current_features(cf)
    , current_descriptors(cd)
  {}

  /// Compute matching track pairs for pending merge
  track_pairs
  match( frame_id_t target_frame )
  {
    // extract the subset of tracks on the target frame
    track_set_sptr tgt_trks = all_tracks->active_tracks(target_frame);
    // extract the set of features on the target frame
    feature_set_sptr target_features = tgt_trks->frame_features(target_frame);
    // extract the set of descriptor on the target frame
    descriptor_set_sptr target_descriptors = tgt_trks->frame_descriptors(target_frame);

    // run the matcher algorithm between the target and current frames
    match_set_sptr mset = matcher->match(target_features, target_descriptors,
                                         current_features, current_descriptors);

    // populate matched track pairs
    std::vector<vital::track_sptr> target_tracks = tgt_trks->tracks();
    std::vector<vital::match> matches = mset->matches();

    track_pairs track_matches;
    for( unsigned i = 0; i < matches.size(); i++ )
    {
      unsigned tgt_idx = matches[i].first;
      unsigned cur_idx = matches[i].second;
      track_matches.push_back(std::make_pair(target_tracks[ tgt_idx ],
                                             current_tracks[ cur_idx ]));
    }

    return track_matches;
  }

private:
  // the feature matcher instance
  vital::algo::match_features_sptr matcher;

  // the set of all tracks to match between
  track_set_sptr all_tracks;

  // the extracted subset of tracks for the current frame
  std::vector<track_sptr> current_tracks;

  // the extracted features from the tracks on the current frame
  feature_set_sptr current_features;

  // the extracted descriptors from the tracks on the current frame
  descriptor_set_sptr current_descriptors;
};
}


/// Private implementation class
class close_loops_keyframe::priv
{
public:
  /// Constructor
  priv()
    : match_req(100),
      search_bandwidth(10),
      min_keyframe_misses(5),
      stop_after_match(false),
      m_logger( vital::get_logger( "arrows.core.close_loops_keyframe" ))
  {
  }

  priv(const priv& other)
    : match_req(other.match_req),
      search_bandwidth(other.search_bandwidth),
      min_keyframe_misses(other.min_keyframe_misses),
      stop_after_match(other.stop_after_match),
      matcher(!other.matcher ? algo::match_features_sptr() : other.matcher->clone()),
      m_logger( vital::get_logger( "arrows.core.close_loops_keyframe" ))
  {
  }

  /// Stich the current frame to the specified target frame number
  int merge_tracks( track_pairs const& matches,
                    track_set_sptr& all_tracks,
                    std::map<track_sptr, track_sptr>& track_replacement ) const
  {
    std::set<vital::track_id_t> to_remove;

    // merge the tracks
    VITAL_FOREACH( auto match, matches )
    {
      track_sptr t1 = match.first;
      track_sptr t2 = match.second;
      // if t2 has already been merged, look-up the new merged track
      auto itr = track_replacement.find(t2);
      if (itr != track_replacement.end())
      {
        t2 = itr->second;
      }
      if( t1->append( *t2 ) )
      {
        // mark this track id for later removal
        to_remove.insert( t2->id() );
        // update the look up table for merged track replacement
        track_replacement[ match.second ] = t1;
        if( t2 != match.second )
        {
          track_replacement[ t2 ] = t1;
        }
      }
    }

    // remove tracks which have been merged
    if( !to_remove.empty() )
    {
      std::vector<track_sptr> at = all_tracks->tracks();
      at.erase(
        std::remove_if( at.begin(), at.end(),
                        std::bind( track_in_set, std::placeholders::_1, &to_remove ) ),
        at.end()
      );
      // recreate the track set with the new filtered tracks
      all_tracks = std::make_shared<simple_track_set>( at );
    }
    return static_cast<int>(to_remove.size());
  }

  /// number of feature matches required for acceptance
  int match_req;

  /// number of adjacent frames to match
  int search_bandwidth;

  /// minimum number of keyframe misses before creating a new keyframe
  unsigned int min_keyframe_misses;

  /// stop matching against additional keyframes if at least one succeeds
  bool stop_after_match;

  /// Indices of the the selected keyframes
  std::vector<frame_id_t> keyframes;

  /// histogram of matches associated with each frame
  std::map<frame_id_t, unsigned int> frame_matches;

  /// a collection of recent frame that didn't match any keyframe
  std::vector<frame_id_t> keyframe_misses;

  /// The feature matching algorithm to use
  vital::algo::match_features_sptr matcher;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
close_loops_keyframe
::close_loops_keyframe()
: d_(new priv)
{
}


/// Copy Constructor
close_loops_keyframe
::close_loops_keyframe(const close_loops_keyframe& other)
: d_(new priv(*other.d_))
{
}


/// Destructor
close_loops_keyframe
::~close_loops_keyframe() VITAL_NOTHROW
{
}


std::string
close_loops_keyframe
::description() const
{
  return "Establishes keyframes matches to all keyframes";
}


/// Get this alg's \link vital::config_block configuration block \endlink
  vital::config_block_sptr
close_loops_keyframe
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Matcher algorithm
  algo::match_features::get_nested_algo_configuration("feature_matcher", config, d_->matcher);

  config->set_value("match_req", d_->match_req,
                    "The required number of features needed to be matched for a success.");

  config->set_value("search_bandwidth", d_->search_bandwidth,
                    "number of adjacent frames to match to (must be at least 1)");

  config->set_value("min_keyframe_misses", d_->min_keyframe_misses,
                    "minimum number of keyframe match misses before creating a new keyframe. "
                    "A match miss occures when the current frame does not match any existing "
                    "keyframe (must be at least 1)");

  config->set_value("stop_after_match", d_->stop_after_match,
                    "If set, stop matching additional keyframes after at least "
                    "one match is found and then one fails to match.  This "
                    "prevents making many comparions to keyframes that are "
                    "likely to fail, but it also misses unexpected matches "
                    "that could make the tracks stronger.");

  return config;
}


/// Set this algo's properties via a config block
void
close_loops_keyframe
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm configuration
  algo::match_features::set_nested_algo_configuration("feature_matcher",
                                                      config, d_->matcher);

  d_->match_req = config->get_value<int>("match_req");
  d_->search_bandwidth = config->get_value<int>("search_bandwidth");
  d_->min_keyframe_misses = config->get_value<unsigned int>("min_keyframe_misses");
  d_->stop_after_match = config->get_value<bool>("stop_after_match");
}


bool
close_loops_keyframe
::check_configuration(vital::config_block_sptr config) const
{

  return (
    algo::match_features::check_nested_algo_configuration("feature_matcher", config)
    && config->get_value<int>("search_bandwidth") >= 1
    && config->get_value<int>("min_keyframe_misses") >= 1
  );
}


/// Frame stitching using keyframe-base matching
vital::track_set_sptr
close_loops_keyframe
::stitch( vital::frame_id_t frame_number,
          vital::track_set_sptr input,
          vital::image_container_sptr,
          vital::image_container_sptr ) const
{
  // initialize frame matches for this frame
  d_->frame_matches[frame_number] = 0;

  // do nothing for the first two frames, there is nothing to match
  if( frame_number < 2 )
  {
    return input;
  }

  // compute the last frame we need to match to within the search bandwidth
  // the conditional accounts for the boundary case at startup
  frame_id_t last_frame = 0;
  if(frame_number > d_->search_bandwidth)
  {
    last_frame = frame_number - d_->search_bandwidth;
  }

  // the first frame is always a key frame (for now)
  // This could proably be improved
  if(d_->keyframes.empty())
  {
    d_->keyframes.push_back(input->first_frame());
  }

  // extract the subset of tracks on the current frame and their
  // features and descriptors
  vital::track_set_sptr current_set = input->active_tracks( frame_number );
  std::vector<vital::track_sptr> current_tracks = current_set->tracks();
  vital::descriptor_set_sptr current_descriptors =
      current_set->frame_descriptors( frame_number );
  vital::feature_set_sptr current_features =
      current_set->frame_features( frame_number );

  track_matcher tmatcher(d_->matcher, input, current_tracks,
                         current_features, current_descriptors);

  // Initialize frame_matches to the number of tracks already matched
  // between the current and previous frames.  This matching was done outside
  // of loop closure as part of the standard frame-to-frame tracking
  d_->frame_matches[frame_number] =
      static_cast<unsigned int>(current_set->active_tracks( frame_number-1 )->size());

  // used to compute the maximum number of matches between the current frame
  // and any of the key frames
  int max_keyframe_matched = 0;

  // use this iterator to step backward through the keyframes
  // as we step backward through the neighborhood to identify
  // which neighborhood frames are also keyframes
  auto kitr = d_->keyframes.rbegin();
  // since loop closure starts at frame n-2, if the latest
  // keyframe happens to be n-1 we need to skip that one
  if (*kitr == frame_number-1)
  {
    ++kitr;
  }

  // start a thread to run matching for each frame of interest
  typedef std::packaged_task<track_pairs(frame_id_t)> match_task_t;
  std::vector<std::thread> threads;
  std::map<vital::frame_id_t, std::future<track_pairs> > all_matches;
  // stitch with all frames within a neighborhood of the current frame
  for(vital::frame_id_t f = frame_number - 2; f >= last_frame; f-- )
  {
    match_task_t task(std::bind(&track_matcher::match, tmatcher,
                                std::placeholders::_1));
    all_matches[f] = task.get_future();
    threads.push_back(std::thread(std::move(task), f));
  }
  // stitch with all previous keyframes
  for(auto kitr = d_->keyframes.rbegin(); kitr != d_->keyframes.rend(); ++kitr)
  {
    // if this frame was already matched above then skip it
    if(*kitr >= last_frame)
    {
      continue;
    }
    match_task_t task(std::bind(&track_matcher::match, tmatcher,
                                std::placeholders::_1));
    all_matches[*kitr] = task.get_future();
    threads.push_back(std::thread(std::move(task), *kitr));
  }

  VITAL_FOREACH(auto& t, threads)
  {
    t.join();
  }


  std::map<track_sptr, track_sptr> track_replacement;
  // stitch with all frames within a neighborhood of the current frame
  for(vital::frame_id_t f = frame_number - 2; f >= last_frame; f-- )
  {
    auto const& matches = all_matches[f].get();
    int num_matched = static_cast<int>(matches.size());
    int num_linked = 0;
    if( num_matched >= d_->match_req )
    {
      num_linked = d_->merge_tracks(matches, input, track_replacement);
    }
    // accumulate matches to help assign keyframes later
    d_->frame_matches[frame_number] += num_matched;

    // keyframes can occur within the current search neighborhood
    // if this frame is a keyframe then account for it in the
    // computation of the maximum number of matches to all keyframes
    std::string frame_name = "";
    if(kitr != d_->keyframes.rend() && f == *kitr)
    {
      if( num_matched > max_keyframe_matched )
      {
        max_keyframe_matched = num_matched;
      }
      ++kitr;
      frame_name = "keyframe ";
    }
    LOG_INFO(d_->m_logger, "Matching frame " << frame_number << " to "
                           << frame_name << f
                           << " has "<< num_matched << " matches and "
                           << num_linked << " joined tracks");
  }
  // divide by number of matched frames to get the average
  d_->frame_matches[frame_number] /=
    static_cast<unsigned int>(frame_number - last_frame);

  // stitch with all previous keyframes
  for(auto kitr = d_->keyframes.rbegin(); kitr != d_->keyframes.rend(); ++kitr)
  {
    // if this frame was already matched above then skip it
    if(*kitr >= last_frame)
    {
      continue;
    }
    auto const& matches = all_matches[*kitr].get();
    int num_matched = static_cast<int>(matches.size());
    int num_linked = 0;
    if( num_matched >= d_->match_req )
    {
      num_linked = d_->merge_tracks(matches, input, track_replacement);
    }
    LOG_INFO(d_->m_logger, "Matching frame " << frame_number << " to keyframe "<< *kitr
                           << " has "<< num_matched << " matches and "
                           << num_linked << " joined tracks");
    if( num_matched > max_keyframe_matched )
    {
      max_keyframe_matched = num_matched;
    }
    // if the stop-after-match option is set and we've already matched a keyframe
    // but this key frame did not match, then exit the loop early and don't
    // match any more key frames.
    if (d_->stop_after_match &&
        max_keyframe_matched >= d_->match_req &&
        num_matched < d_->match_req)
    {
      break;
    }
  }

  // keep track of frames that matched no keyframes
  if (max_keyframe_matched < d_->match_req)
  {
    d_->keyframe_misses.push_back(frame_number);
    LOG_DEBUG(d_->m_logger, "Frame " << frame_number << " added to keyframe misses. "
                            << "Count: "<<d_->keyframe_misses.size());
  }

  // If we've seen enough keyframe misses and the first miss has passed outside
  // of the search bandwidth, then add a new key frame by selecting the frame
  // since the first miss that has been most successful at matching.
  if (d_->keyframe_misses.size() > d_->min_keyframe_misses &&
      d_->keyframe_misses.front() < last_frame)
  {
    auto hitr = d_->frame_matches.find(d_->keyframe_misses.front());
    unsigned int max_matches = 0;
    frame_id_t max_id = d_->keyframe_misses.front();
    // find the frame with the most accumulated matches
    for(++hitr; hitr != d_->frame_matches.end(); ++hitr)
    {
      if(hitr->second > max_matches)
      {
        max_matches = hitr->second;
        max_id = hitr->first;
      }
    }
    // the new key frame must have the required number of matches on average
    if( max_matches > static_cast<unsigned int>(d_->match_req) )
    {
      // create the new keyframe and clear the list of misses
      LOG_INFO(d_->m_logger, "creating new keyframe on frame " << max_id);
      d_->keyframes.push_back(max_id);
      d_->keyframe_misses.clear();
    }
  }

  return input;
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
