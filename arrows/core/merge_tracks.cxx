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

/**
 * \file
 * \brief Implementation of functions to match and merge tracks
 */

#include "merge_tracks.h"


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;



/// Compute matching track pairs between two frames
track_pairs_t
match_tracks( vital::algo::match_features_sptr matcher,
              vital::feature_track_set_sptr all_tracks,
              vital::frame_id_t current_frame,
              vital::frame_id_t target_frame )
{
  // extract the subset of tracks on the current frame
  auto current_tracks = std::make_shared<feature_track_set>(
                            all_tracks->active_tracks(current_frame) );
  // extract the set of features on the current frame
  feature_set_sptr current_features = current_tracks->frame_features(current_frame);
  // extract the set of descriptor on the current frame
  descriptor_set_sptr current_descriptors = current_tracks->frame_descriptors(current_frame);

  return match_tracks(matcher, all_tracks,
                      current_tracks, current_features, current_descriptors,
                      target_frame);
}


/// Compute matching track pairs between two frames
track_pairs_t
match_tracks( vital::algo::match_features_sptr matcher,
              vital::feature_track_set_sptr all_tracks,
              vital::feature_track_set_sptr current_tracks,
              vital::feature_set_sptr current_features,
              vital::descriptor_set_sptr current_descriptors,
              vital::frame_id_t target_frame )
{
  // extract the subset of tracks on the target frame
  auto target_tracks = std::make_shared<feature_track_set>(
                           all_tracks->active_tracks(target_frame) );
  // extract the set of features on the target frame
  feature_set_sptr target_features = target_tracks->frame_features(target_frame);
  // extract the set of descriptor on the target frame
  descriptor_set_sptr target_descriptors = target_tracks->frame_descriptors(target_frame);

  return match_tracks(matcher,
                      current_tracks, current_features, current_descriptors,
                      target_tracks, target_features, target_descriptors);
}


/// Compute matching track pairs between two frames
track_pairs_t
match_tracks( vital::algo::match_features_sptr matcher,
              vital::feature_track_set_sptr current_tracks,
              vital::feature_set_sptr current_features,
              vital::descriptor_set_sptr current_descriptors,
              vital::feature_track_set_sptr target_tracks,
              vital::feature_set_sptr target_features,
              vital::descriptor_set_sptr target_descriptors)
{
  // run the matcher algorithm between the target and current frames
  match_set_sptr mset = matcher->match(target_features, target_descriptors,
                                       current_features, current_descriptors);
  if( !mset )
  {
    return track_pairs_t();
  }

  // populate matched track pairs
  std::vector<vital::track_sptr> cur_tracks = current_tracks->tracks();
  std::vector<vital::track_sptr> tgt_tracks = target_tracks->tracks();
  std::vector<vital::match> matches = mset->matches();

  track_pairs_t track_matches;
  for( unsigned i = 0; i < matches.size(); i++ )
  {
    unsigned tgt_idx = matches[i].first;
    unsigned cur_idx = matches[i].second;
    track_matches.push_back(std::make_pair(cur_tracks[ cur_idx ],
                                           tgt_tracks[ tgt_idx ]));
  }

  return track_matches;
}


/// Merge the pairs of tracks provided by \p matches if possible
int
merge_tracks( track_pairs_t const& matches,
              track_map_t& track_replacement )
{
  int num_merged = 0;

  // merge the tracks
  for( auto match : matches )
  {
    track_sptr t1 = match.first;
    track_sptr t2 = match.second;
    // if t1 has already been merged, look-up the new merged track
    auto itr = track_replacement.find(t1);
    if (itr != track_replacement.end())
    {
      t1 = itr->second;
    }
    if( t2->append( *t1 ) )
    {
      ++num_merged;
      // update the look up table for merged track replacement
      track_replacement[ match.first ] = t2;
      if( t1 != match.first )
      {
        track_replacement[ t1 ] = t2;
      }
    }
  }
  return num_merged;
}


/// Remove all track with keys in the \p track_replacement map
feature_track_set_sptr
remove_replaced_tracks( vital::feature_track_set_sptr all_tracks,
                        track_map_t const& track_replacement )
{
  if( track_replacement.empty() )
  {
    return all_tracks;
  }

  // remove tracks which have been replaced by the merger into another track
  std::vector<track_sptr> at = all_tracks->tracks();
  at.erase(
    std::remove_if( at.begin(), at.end(), [&track_replacement] ( track_sptr t )
      {
        return track_replacement.find( t ) != track_replacement.end();
      }
    ),
    at.end()
  );
  // recreate the track set with the new filtered tracks
  return std::make_shared<feature_track_set>( at );
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
