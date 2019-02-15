/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief close_loops_appearance_indexed algorithm implementation
 */

#include <map>
#include <algorithm>
#include <iterator>
#include <limits>

#include "close_loops_appearance_indexed.h"


using namespace kwiver::vital;

#include <vital/logger/logger.h>
#include <vital/algo/algorithm.h>
#include <vital/algo/match_descriptor_sets.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/image_io.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_fundamental_matrix.h>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace arrows {
namespace core {

class close_loops_appearance_indexed::priv
{
public:
  priv();

  typedef std::pair<feature_track_state_sptr, feature_track_state_sptr> fs_match;
  typedef std::vector<fs_match> matches_vec;

  kwiver::vital::feature_track_set_sptr
  detect(
    kwiver::vital::feature_track_set_sptr feat_tracks,
    kwiver::vital::frame_id_t frame_number);

  kwiver::vital::feature_track_set_sptr
  verify_and_add_image_matches(
    kwiver::vital::feature_track_set_sptr feat_tracks,
    kwiver::vital::frame_id_t frame_number,
    std::vector<frame_id_t> const &putative_matches);

  void
  do_matching(
    const std::vector<feature_track_state_sptr> &va,
    const std::vector<feature_track_state_sptr> &vb,
    matches_vec &matches);

  kwiver::vital::feature_track_set_sptr
  verify_and_add_image_matches_node_id_guided(
    kwiver::vital::feature_track_set_sptr feat_tracks,
    kwiver::vital::frame_id_t frame_number,
    std::vector<frame_id_t> const &putative_matches);

  typedef std::map<unsigned int, std::vector<feature_track_state_sptr>> node_id_to_feat_map;

  node_id_to_feat_map make_node_map(const std::vector<feature_track_state_sptr> &feats);

  match_set_sptr
  remove_duplicate_matches(
    match_set_sptr mset,
    feature_info_sptr fi1,
    feature_info_sptr fi2);

  /// The logger handle
  kwiver::vital::logger_handle_t m_logger;

  /// The feature matching algorithm to use
  vital::algo::match_features_sptr m_matcher;

  /// The bag of words matching image finder
  vital::algo::match_descriptor_sets_sptr m_bow;

  /// The fundamental matrix estimator for geometric verification
  vital::algo::estimate_fundamental_matrix_sptr m_f_estimator;

  /// The minimum number of inliers required for a putative loop to be accepted
  unsigned m_min_loop_inlier_matches;

  /// The function used to calculate the distance between two descriptors
  std::function<float(descriptor_sptr, descriptor_sptr)> desc_dist = hamming_distance;

  /// Inlier threshold for fundamental matrix geometric verification
  double m_geometric_verification_inlier_threshold;

  /// The maximum number of times to attempt to complete a loop with each new frame
  int m_max_loop_attempts_per_frame;

  // If this or more tracks ids are shared between two frames then don't attempt to close the loop
  int m_tracks_in_common_to_skip_loop_closing;

  //if intersect over union of track ids between two frames are greather than this then don't try to close the loop.
  float m_skip_loop_detection_track_i_over_u_threshold;

  // Must have this inlier fraction to accept a loop completion
  float m_min_loop_inlier_fraction;
};

//-----------------------------------------------------------------------------

close_loops_appearance_indexed::priv
::priv()
  : m_f_estimator(),
  m_min_loop_inlier_matches(128),
  m_geometric_verification_inlier_threshold(2.0),
  m_max_loop_attempts_per_frame(200),
  m_tracks_in_common_to_skip_loop_closing(0),
  m_skip_loop_detection_track_i_over_u_threshold(0.5),
  m_min_loop_inlier_fraction(0.5)
{
}

//-----------------------------------------------------------------------------

close_loops_appearance_indexed::priv::node_id_to_feat_map
close_loops_appearance_indexed::priv
::make_node_map(const std::vector<feature_track_state_sptr> &feats)
{
  node_id_to_feat_map fm;
  for (auto f : feats)
  {
    if (f->descriptor && f->descriptor->node_id() != std::numeric_limits<unsigned int>::max())
    {
      auto node_id = f->descriptor->node_id();
      auto fm_it = fm.find(node_id);
      if (fm_it != fm.end())
      {
        fm_it->second.push_back(f);
      }
      else
      {
        std::vector<feature_track_state_sptr> vec;
        vec.push_back(f);
        fm[node_id] = vec;
      }
    }
  }
  return fm;
}

void
close_loops_appearance_indexed::priv
::do_matching(const std::vector<feature_track_state_sptr> &va,
              const std::vector<feature_track_state_sptr> &vb,
              matches_vec & matches)
{
  const int max_int = std::numeric_limits<int>::max();
  const int match_thresh = 128;
  const float next_neigh_match_diff = 1.2;


  std::map<track_id_t, feature_track_state_sptr> track_to_vb_state;

  // store mapping from track id to feature track state from vb
  for (auto match_feat : vb)
  {
    track_to_vb_state[match_feat->track()->id()] = match_feat;
  }

  //now loop over all the features in the same bin
  for (auto cur_feat : va)
  {
    int dist1 = max_int;
    int dist2 = max_int;
    vital::feature_track_state_sptr best_match = nullptr;

    // see if this track id already has a vb feature track state associate with it
    auto it = track_to_vb_state.find(cur_feat->track()->id());
    if(it != track_to_vb_state.end())
    {
        // The two features are already from the same track.  So, they are
        // a match.  Add them to matches.
        matches.push_back(fs_match(cur_feat, it->second));
        // There is no need to search vb for additional matches.
        continue;
    }

    for (auto match_feat : vb)
    {
      int dist = static_cast<int>(desc_dist(cur_feat->descriptor, match_feat->descriptor));
      if (dist < dist1)
      {
        dist1 = dist;
        best_match = match_feat;
      }
      else if (dist < dist2)
      {
        dist2 = dist;
      }
    }
    if (best_match && dist1 != max_int && dist2 != max_int)
    {
      if (dist1 < match_thresh  && dist2 > next_neigh_match_diff * dist1)
      {
        matches.push_back(fs_match(cur_feat, best_match));
      }
    }
  }
}


kwiver::vital::feature_track_set_sptr
close_loops_appearance_indexed::priv
::verify_and_add_image_matches_node_id_guided(
  kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number,
  std::vector<frame_id_t> const &putative_matches)
{
  auto cur_frame_fts = feat_tracks->frame_feature_track_states(frame_number);

  auto cur_frame_track_ids = feat_tracks->active_track_ids(frame_number);

  int num_successfully_matched_pairs = 0;

  auto cur_node_map = make_node_map(cur_frame_fts);
  int num_failed_loop_attempts_in_a_row = 0;
  //loop over putatively matching frames
  for (auto fn_match : putative_matches)
  {

    if (fn_match == frame_number)
    {
      continue; // no sense matching an image to itself
    }

    //get active tracks on fn match
    auto match_frame_track_ids = feat_tracks->active_track_ids(fn_match);
    std::set<track_id_t> tracks_in_common, union_of_tracks;
    std::set_intersection(cur_frame_track_ids.begin(), cur_frame_track_ids.end(),
                          match_frame_track_ids.begin(), match_frame_track_ids.end(),
                          std::inserter(tracks_in_common, tracks_in_common.begin()));

    std::set_union(cur_frame_track_ids.begin(), cur_frame_track_ids.end(),
                   match_frame_track_ids.begin(), match_frame_track_ids.end(),
                   std::inserter(union_of_tracks, union_of_tracks.begin()));

    double i_over_u = static_cast<double>(tracks_in_common.size()) /
                      static_cast<double>(union_of_tracks.size());

    if (i_over_u > m_skip_loop_detection_track_i_over_u_threshold)
    {
      continue;
    }

    // how many tracks to fn_match and frame_number have in common?  Too many?  Don't match.

    ++num_failed_loop_attempts_in_a_row;
    if (num_failed_loop_attempts_in_a_row > m_max_loop_attempts_per_frame)
    {
      break;
    }

    auto match_frame_fts = feat_tracks->frame_feature_track_states(fn_match);

    auto match_node_map = make_node_map(match_frame_fts);
    matches_vec validated_matches;

    //ok now we do the matching.
    //loop over node ids in the current frame
    for (auto fc : cur_node_map)
    {
      //find the matching node id in the putative matching frame
      auto node_id = fc.first;
      auto fm = match_node_map.find(node_id);
      if (fm == match_node_map.end())
      {
        //no features with the same node in the putative matching frame
        continue;
      }

      auto &cur_feat_vec = fc.second;
      auto &match_feat_vec = fm->second;
      matches_vec matches_forward, matches_reverse;

      do_matching( cur_feat_vec, match_feat_vec, matches_forward);
      if (matches_forward.empty())
      {
        continue;
      }
      do_matching( match_feat_vec, cur_feat_vec, matches_reverse);
      if (matches_reverse.empty())
      {
        continue;
      }
      // cross-validate the matches
      for (auto m_f : matches_forward)
      {
        for (auto m_r : matches_reverse)
        {
          if (m_f.first == m_r.second && m_f.second == m_r.first)
          {
            validated_matches.push_back(m_f);
            break;
          }
        }
      }
    }

    if (validated_matches.size() < m_min_loop_inlier_matches)
    {
      continue;
    }

    size_t already_joined_matches = 0;
    for (auto &vm : validated_matches)
    {
      if (vm.first->track()->id() == vm.second->track()->id())
      {
        ++already_joined_matches;
      }
    }

    if (already_joined_matches == validated_matches.size())
    {
      num_failed_loop_attempts_in_a_row = 0;
      continue;
    }

    std::vector<bool> inliers;
    //do geometric verification here
    if (m_f_estimator)
    {
      std::vector<vector_2d> pts_right, pts_left;
      for (auto &m : validated_matches)
      {
        pts_right.push_back(m.first->feature->loc());
        pts_left.push_back(m.second->feature->loc());
      }

      auto F = m_f_estimator->estimate(pts_right, pts_left, inliers,
                              m_geometric_verification_inlier_threshold);

      if (!F)
      {
        continue;
      }

      unsigned num_inliers =
        static_cast<unsigned>(std::count(inliers.begin(), inliers.end(), true));

      float inlier_fraction = static_cast<double>(num_inliers) / static_cast<double>(validated_matches.size());

      if (num_inliers < m_min_loop_inlier_matches || inlier_fraction < m_min_loop_inlier_fraction)
      {
        continue;
      }
    }

    num_failed_loop_attempts_in_a_row = 0;

    int num_stitched_tracks = 0;
    for(size_t i = 0; i < validated_matches.size(); ++i)
    {
      if (!inliers.empty())
      {
        if (!inliers[i])
        {
          continue;
        }
      }
      auto &m = validated_matches[i];
      track_sptr t1 = m.first->track();
      track_sptr t2 = m.second->track();
      //t1's states should be after t2
      if (t1->last_frame() < t2->last_frame())
      {
        std::swap(t1, t2);
      }

      if (feat_tracks->merge_tracks(t1, t2))
      {
        //tracks will not merge if t1 and t2 are already the same track
        ++num_stitched_tracks;
      }
    }

    if (num_stitched_tracks > 0)
    {
      LOG_DEBUG(m_logger, "Stitched " << num_stitched_tracks <<
        " tracks between frames " << frame_number <<
        " and " << fn_match);

      ++num_successfully_matched_pairs;
    }
  }

  LOG_DEBUG(m_logger, "Of " << putative_matches.size() << " putative matches "
    << num_successfully_matched_pairs << " pairs were verified");

  return feat_tracks;
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
close_loops_appearance_indexed::priv
::verify_and_add_image_matches(
  kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number,
  std::vector<frame_id_t> const &putative_matches)
{
  feature_set_sptr feat1, feat2;
  descriptor_set_sptr desc1, desc2;

  feature_info_sptr fi1 = feat_tracks->frame_feature_info(frame_number,true);
  feat1 = fi1->features;
  desc1 = fi1->descriptors;

  int num_successfully_matched_pairs = 0;

  for (auto fn2 : putative_matches)
  {
    if (fn2 == frame_number)
    {
      continue; // no sense matching an image to itself
    }
    feature_info_sptr fi2 = feat_tracks->frame_feature_info(fn2, true);
    feat2 = fi2->features;
    desc2 = fi2->descriptors;

    match_set_sptr mset = m_matcher->match(feat1, desc1, feat2, desc2);
    if (!mset)
    {
      LOG_WARN(m_logger, "Feature matching between frames " << frame_number <<
        " and " << fn2 << " failed");
      continue;
    }

    mset = remove_duplicate_matches(mset, fi1, fi2);

    std::vector<match> vm = mset->matches();

    if (vm.size() < m_min_loop_inlier_matches)
    {
      continue;
    }

    int num_linked = 0;
    for (match m : vm)
    {
      track_sptr t1 = fi1->corresponding_tracks[m.first];
      track_sptr t2 = fi2->corresponding_tracks[m.second];
      if (feat_tracks->merge_tracks(t1, t2))
      {
        ++num_linked;
      }
    }
    LOG_DEBUG(m_logger, "Stitched " << num_linked <<
      " tracks between frames " << frame_number <<
      " and " << fn2);

    if (num_linked > 0)
    {
      ++num_successfully_matched_pairs;
    }
  }

  LOG_DEBUG(m_logger, "Of " << putative_matches.size() << " putative matches "
    << num_successfully_matched_pairs << " pairs were verified");

  return feat_tracks;
}

//-----------------------------------------------------------------------------

match_set_sptr
close_loops_appearance_indexed::priv
::remove_duplicate_matches(match_set_sptr mset,
                           feature_info_sptr fi1,
                           feature_info_sptr fi2)
{
  std::vector<match> orig_matches = mset->matches();

  struct match_with_cost {
    unsigned m1;
    unsigned m2;
    double cost;

    match_with_cost()
      : m1(0)
      , m2(0)
      , cost(std::numeric_limits<double>::infinity())
    {}

    match_with_cost(unsigned _m1, unsigned _m2, double _cost)
      :m1(_m1)
      , m2(_m2)
      , cost(_cost)
    {}

    bool operator<(const match_with_cost &rhs) const
    {
      return cost < rhs.cost;
    }
  };

  std::vector<match_with_cost> mwc_vec;
  mwc_vec.resize(orig_matches.size());
  size_t m_idx = 0;
  auto fi1_features = fi1->features->features();
  auto fi2_features = fi2->features->features();
  for (auto m : orig_matches)
  {
    match_with_cost & mwc = mwc_vec[m_idx++];
    mwc.m1 = m.first;
    mwc.m2 = m.second;
    feature_sptr f1 = fi1_features[mwc.m1];
    feature_sptr f2 = fi2_features[mwc.m2];
    //using relative scale as cost
    mwc.cost = f1->scale() / f2->scale();
  }
  std::sort(mwc_vec.begin(), mwc_vec.end());
  //now get the median cost (scale change)
  double median_cost = mwc_vec[mwc_vec.size() / 2].cost;
  // now adjust the costs according to how different they are from the median cost.
  // Adjusting to the median cost accounts for changes in zoom.  If most matches have a
  // scale change, then that should be the low cost option to pick when sorting.
  for (auto &m : mwc_vec)
  {
    m.cost /= median_cost;
    m.cost = std::max(m.cost, 1.0 / m.cost);
  }

  //sort again.  This time with the median adjusted costs.
  std::sort(mwc_vec.begin(), mwc_vec.end());

  // sorting makes us add the lowest cost matches first.  This means if we have
  // duplicate matches, the best ones will end up in the final match set.
  std::set<unsigned> matched_indices_1, matched_indices_2;

  std::vector<match> unique_matches;

  for (auto m : mwc_vec)
  {
    if (matched_indices_1.find(m.m1) != matched_indices_1.end() ||
        matched_indices_2.find(m.m2) != matched_indices_2.end())
    {
      continue;
    }
    matched_indices_1.insert(m.m1);
    matched_indices_2.insert(m.m2);
    unique_matches.push_back(vital::match(m.m1,m.m2));
  }

  return std::make_shared<simple_match_set>(unique_matches);

}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
close_loops_appearance_indexed::priv
::detect(kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number)
{
  if (!m_bow)
  {
    return feat_tracks;
  }

  std::vector<frame_id_t> putative_matching_images;

  auto fd = std::dynamic_pointer_cast<feature_track_set_frame_data>(feat_tracks->frame_data(frame_number));
  if (!fd || !fd->is_keyframe)
  {
    //not a keyframe so just try to match to the last few frames
    auto all_fids = feat_tracks->all_frame_ids();
    for (auto it = all_fids.rbegin(); it != all_fids.rend(); ++it)
    {
      if (*it != frame_number)
      {
        putative_matching_images.push_back(*it);
      }
      if (putative_matching_images.size() >= 5)
      {
        break;
      }
    }
  }
  else
  {
    auto desc = feat_tracks->frame_descriptors(frame_number);

    putative_matching_images =
      m_bow->query_and_append(desc, frame_number);
  }
  return verify_and_add_image_matches_node_id_guided(feat_tracks, frame_number,
                                      putative_matching_images);
}

// ----------------------------------------------------------------------------

close_loops_appearance_indexed
::close_loops_appearance_indexed()
{
  d_ = std::make_shared<priv>();
  attach_logger("arrows.core.close_loops_appearance_indexed");
  d_->m_logger = this->logger();
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
close_loops_appearance_indexed
::stitch(kwiver::vital::frame_id_t frame_number,
  kwiver::vital::feature_track_set_sptr input,
  kwiver::vital::image_container_sptr image,
  kwiver::vital::image_container_sptr mask) const
{
  return d_->detect(input, frame_number);
}

//-----------------------------------------------------------------------------

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
close_loops_appearance_indexed
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm

  algo::match_features::
    get_nested_algo_configuration("match_features", config, d_->m_matcher);

  algo::match_descriptor_sets::
    get_nested_algo_configuration("bag_of_words_matching", config, d_->m_bow);

  // nested algorithm configurations
  vital::algo::estimate_fundamental_matrix
    ::get_nested_algo_configuration("fundamental_mat_estimator",
      config, d_->m_f_estimator);

  config->set_value("min_loop_inlier_matches",
    d_->m_min_loop_inlier_matches,
    "the minimum number of inlier feature matches to accept a loop connection and join tracks");

  config->set_value("geometric_verification_inlier_threshold",
    d_->m_geometric_verification_inlier_threshold,
    "inlier threshold for fundamental matrix based geometric verification of loop closure in pixels");

  config->set_value("max_loop_attempts_per_frame",
    d_->m_max_loop_attempts_per_frame,
    "the maximum number of loop closure attempts to make per frame");

  config->set_value("tracks_in_common_to_skip_loop_closing",
    d_->m_tracks_in_common_to_skip_loop_closing,
    "if this or more tracks are in common between two frames then don't try to complete a loop with them");

  config->set_value("m_skip_loop_detection_track_i_over_u_threshold",
    d_->m_skip_loop_detection_track_i_over_u_threshold,
    "skip loop detection if intersection over union of track ids in two frames is greater than this");

  config->set_value("min_loop_inlier_fraction",
    d_->m_min_loop_inlier_fraction,
    "inlier fraction must be this high to accept a loop completion");

  return config;
}

//-----------------------------------------------------------------------------

/// Set this algo's properties via a config block
void
close_loops_appearance_indexed
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values
  // are present.  An alternative is to check for key presence before
  // performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.

  algo::match_features_sptr mf;
  algo::match_features::set_nested_algo_configuration(
    "match_features", config, mf);
  d_->m_matcher = mf;

  algo::match_descriptor_sets_sptr bow;
  algo::match_descriptor_sets::set_nested_algo_configuration(
    "bag_of_words_matching", config, bow);
  d_->m_bow = bow;

  vital::algo::estimate_fundamental_matrix
    ::set_nested_algo_configuration("fundamental_mat_estimator",
      config, d_->m_f_estimator);

  d_->m_min_loop_inlier_matches =
    config->get_value<int>("min_loop_inlier_matches",
      d_->m_min_loop_inlier_matches);

  d_->m_geometric_verification_inlier_threshold =
    config->get_value<double>("geometric_verification_inlier_threshold",
      d_->m_geometric_verification_inlier_threshold);

  d_->m_max_loop_attempts_per_frame =
    config->get_value<int>("max_loop_attempts_per_frame",
      d_->m_max_loop_attempts_per_frame);

  d_->m_tracks_in_common_to_skip_loop_closing =
    config->get_value<int>("tracks_in_common_to_skip_loop_closing",
      d_->m_tracks_in_common_to_skip_loop_closing);

  d_->m_skip_loop_detection_track_i_over_u_threshold =
    config->get_value<float>("skip_loop_detection_track_i_over_u_threshold",
      d_->m_skip_loop_detection_track_i_over_u_threshold);

  d_->m_min_loop_inlier_fraction =
    config->get_value<float>("m_min_loop_inlier_fraction",
      d_->m_min_loop_inlier_fraction);
}

//-----------------------------------------------------------------------------

bool
close_loops_appearance_indexed
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid =
    algo::match_features::check_nested_algo_configuration(
      "match_features", config) && config_valid;

  config_valid =
    algo::match_descriptor_sets::check_nested_algo_configuration(
      "bag_of_words_matching", config) && config_valid;

  int min_loop_matches =
    config->get_value<int>("min_loop_inlier_matches");

  if (min_loop_matches < 0)
  {
    LOG_ERROR(d_->m_logger,
      "min_loop_inlier_matches must be non-negative");
    config_valid = false;
  }

  return config_valid;
}

//-----------------------------------------------------------------------------

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
