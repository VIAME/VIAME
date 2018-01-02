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
 * \brief OCV detect_loops algorithm implementation
 */

#include <map>
#include <algorithm>

#include "detect_loops.h"


using namespace kwiver::vital;

#include <DBoW2\DBoW2.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <vital/logger/logger.h>
#include <vital/algo/algorithm.h>
#include <vital/algo/bag_of_words_matching.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/image_io.h>
#include <vital/algo/match_features.h>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace arrows {
namespace ocv {

class detect_loops::priv 
{
public:
  priv();

  kwiver::vital::feature_track_set_sptr
    detect(kwiver::vital::feature_track_set_sptr feat_tracks,
      kwiver::vital::frame_id_t frame_number);

  kwiver::vital::feature_track_set_sptr
    verify_and_add_image_matches(
      kwiver::vital::feature_track_set_sptr feat_tracks,
      kwiver::vital::frame_id_t frame_number,
      std::vector<frame_id_t> const &putative_matches);

  match_set_sptr
    remove_duplicate_matches(
      match_set_sptr mset, 
      feature_info_sptr fi1,
      feature_info_sptr fi2);

  kwiver::vital::logger_handle_t m_logger;
  
  
  /// The feature matching algorithm to use
  vital::algo::match_features_sptr m_matcher;

  // The bag of words matching image finder
  vital::algo::bag_of_words_matching_sptr m_bow;
  
  unsigned m_min_loop_inlier_matches;

};

//-----------------------------------------------------------------------------

detect_loops::priv
::priv()
  : m_min_loop_inlier_matches(50)
{
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
detect_loops::priv
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
detect_loops::priv
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
      , cost(DBL_MAX)
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
  for (auto m : orig_matches)
  {    
    match_with_cost & mwc = mwc_vec[m_idx++];
    mwc.m1 = m.first;
    mwc.m2 = m.second;
    feature_sptr f1 = fi1->features->features()[mwc.m1];
    feature_sptr f2 = fi2->features->features()[mwc.m2];       
    //using relative scale as cost
    mwc.cost = std::max(f1->scale() / f2->scale(), f2->scale() / f1->scale());  
  }

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
detect_loops::priv
::detect(kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number)
{

  std::vector<frame_id_t> putative_matching_images;

  if (!m_bow)
  {
    return feat_tracks;
  } 

  descriptor_set_sptr desc = feat_tracks->frame_descriptors(frame_number);

  m_bow->query(desc, frame_number, putative_matching_images, true);

  return verify_and_add_image_matches(feat_tracks, frame_number, 
                                      putative_matching_images);
}

// ----------------------------------------------------------------------------

detect_loops
::detect_loops()
{
  d_ = std::make_shared<priv>();
  attach_logger("detect_loops");
  d_->m_logger = this->logger();
}

//-----------------------------------------------------------------------------

kwiver::vital::feature_track_set_sptr
detect_loops
::detect(kwiver::vital::feature_track_set_sptr feat_tracks,
  kwiver::vital::frame_id_t frame_number)
{
  return d_->detect(feat_tracks, frame_number);
}

//-----------------------------------------------------------------------------

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_loops
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm

  algo::match_features::
    get_nested_algo_configuration("match_features", config, d_->m_matcher);

  algo::bag_of_words_matching::
    get_nested_algo_configuration("bag_of_words_matching", config, d_->m_bow);

  return config;
}

//-----------------------------------------------------------------------------

/// Set this algo's properties via a config block
void
detect_loops
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

  algo::bag_of_words_matching_sptr bow;
  algo::bag_of_words_matching::set_nested_algo_configuration(
    "bag_of_words_matching", config, bow);
  d_->m_bow = bow;
}

//-----------------------------------------------------------------------------

bool
detect_loops
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid = 
    algo::match_features::check_nested_algo_configuration(
      "match_features", config) && config_valid;

  config_valid =
    algo::bag_of_words_matching::check_nested_algo_configuration(
      "bag_of_words_matching", config) && config_valid;
        
  return config_valid;
}

//-----------------------------------------------------------------------------

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
