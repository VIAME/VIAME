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
 * \brief Header defining functions to match and merge tracks
 */

#ifndef KWIVER_ARROWS_CORE_MERGE_TRACKS_H_
#define KWIVER_ARROWS_CORE_MERGE_TRACKS_H_


#include <map>
#include <vector>

#include <vital/algo/match_features.h>
#include <vital/types/feature_track_set.h>


namespace kwiver {
namespace arrows {
namespace core {

/// Typedef for a vector of pairs of tracks
typedef std::vector<std::pair<vital::track_sptr, vital::track_sptr> > track_pairs_t;
/// Typedef for a map from one track to another
typedef std::map<vital::track_sptr, vital::track_sptr> track_map_t;


/// Compute matching feature track pairs between two frames
/**
 * This function extracts all the feature tracks found on \p current_frame and
 * \p target_frame.  It then extracts the corresponding features and descriptors
 * and uses the provided matcher algorithm to identify matching tracks
 *
 *  \param matcher The matcher algorithm to use in feature/descriptor matching
 *  \param all_tracks The set of all feature tracks on which to detect matches
 *  \param current_frame The index of the source frame to match
 *  \param target_frame The index of the destination frame to match
 *  \return A vector of matching track pairs of the form (current, target)
 */
track_pairs_t match_tracks( vital::algo::match_features_sptr matcher,
                            vital::feature_track_set_sptr all_tracks,
                            vital::frame_id_t current_frame,
                            vital::frame_id_t target_frame );


/// Compute matching feature track pairs between two frames
/**
 * This function extracts all the feature tracks found on \p target_frame.
 * It then extracts the corresponding features and descriptors
 * and uses the provided matcher algorithm to identify matching tracks between
 * the set of provided current tracks, features, and descriptors.  This version
 * of the function exists so that the current tracks, features, and
 * descriptors do not need to be extracted each time if matching against
 * multiple target frames.
 *
 *  \param matcher The matcher algorithm to use in feature/descriptor matching
 *  \param all_tracks The set of all feature tracks on which to detect matches
 *  \param current_tracks A subset of \p all_tracks intersecting the source frame
 *  \param current_features The features corresponding to \p current_tracks on the source frame
 *  \param current_descriptors The descriptors corresponding to \p current_tracks on the source frame
 *  \param target_frame The index of the destination frame to match
 *  \return A vector of matching track pairs of the form (current, target)
 */
track_pairs_t match_tracks( vital::algo::match_features_sptr matcher,
                            vital::feature_track_set_sptr all_tracks,
                            vital::feature_track_set_sptr current_tracks,
                            vital::feature_set_sptr current_features,
                            vital::descriptor_set_sptr current_descriptors,
                            vital::frame_id_t target_frame );


/// Compute matching feature track pairs between two frames
/**
 * This function uses the provide matcher algorithm to identify matching tracks
 * between the sets of provided tracks, features, and descriptors.
 * It is assumed that the current and target track sets contain only tracks
 * with states covering the current and target frames respectively.
 * Furthermore, the provided features and descriptors are extracted from the
 * corresponding tracks on those frames.  This version of the function exists
 * so that the current tracks, features, and descriptors do not need to be
 * extracted each time if matching multiple frame combinations.
 *
 *  \param matcher The matcher algorithm to use in feature/descriptor matching
 *  \param current_tracks A set of feature tracks intersecting the source frame
 *  \param current_features The features corresponding to \p current_tracks on the source frame
 *  \param current_descriptors The descriptors corresponding to \p current_tracks on the source frame
 *  \param target_tracks A set of feature tracks intersecting the target frame
 *  \param target_features The features corresponding to \p target_tracks on the target frame
 *  \param target_descriptors The descriptors corresponding to \p target_tracks on the target frame
 *  \return A vector of matching track pairs of the form (current, target)
 */
track_pairs_t match_tracks( vital::algo::match_features_sptr matcher,
                            vital::feature_track_set_sptr current_tracks,
                            vital::feature_set_sptr current_features,
                            vital::descriptor_set_sptr current_descriptors,
                            vital::feature_track_set_sptr target_tracks,
                            vital::feature_set_sptr target_features,
                            vital::descriptor_set_sptr target_descriptors);


/// Merge the pairs of tracks provided by \p matches if possible
/**
 * For each (t1, t2) pair in \p matches try to merge t1 into t2.
 * Merging copies the track states from t1 into t2 and is only allowed if
 * the tracks do not overlap temporally.  If successful update the
 * \p track_replacement map to indicate which tracks have now been subsumed
 * into which other tracks.
 *
 * \param matches A vector of pairs of tracks to attempt to merge
 * \param track_replacement A map indicating which tracks have been merged
 * \returns the number of successful merges
 */
int merge_tracks( track_pairs_t const& matches,
                  track_map_t& track_replacement );


/// Remove all track with keys in the \p track_replacement map
/**
 * The \p track_replacement is generated by the merge_tracks function.
 * This function takes the set of all tracks and removes those tracks
 * that have been replaced by merging into another track.
 *
 * \param all_tracks The set of all feature tracks to process
 * \param track_replacement The keys of this map are removed from \p all_tracks
 * \return A feature track set in which the replaced tracks have been removed
 */
vital::feature_track_set_sptr
remove_replaced_tracks( vital::feature_track_set_sptr all_tracks,
                        track_map_t const& track_replacement );

} // end namespace core
} // end namespace arrows
} // end namespace kwiver


#endif // KWIVER_ARROWS_CORE_MERGE_TRACKS_H_
