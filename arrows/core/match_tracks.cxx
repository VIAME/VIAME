// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of functions to match tracks
 */

#include "match_tracks.h"

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

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
