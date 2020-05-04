/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
 * \brief Implementation of the core match_features_fundamental_matrix algorithm
 */

#include "match_features_fundamental_matrix.h"

#include <algorithm>
#include <iostream>

#include <vital/exceptions/algorithm.h>
#include <vital/types/fundamental_matrix.h>
#include <vital/types/match_set.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {

namespace core
{

// Private implementation class
class match_features_fundamental_matrix::priv
{
public:
  // Constructor
  priv()
  : inlier_scale(10.0),
    min_required_inlier_count(0),
    min_required_inlier_percent(0.0),
    motion_filter_percentile(0.75)
  {
  }

  // the scale of inlier points
  double inlier_scale;

  // min inlier count required to make any matches
  int min_required_inlier_count;

  // min inlier percent required to make any matches
  double min_required_inlier_percent;

  // filter outliers with motion larger than this percentile
  double motion_filter_percentile;

  // The feature matching algorithm to use
  vital::algo::match_features_sptr matcher_;

  // The fundamental matrix estimation algorithm to use
  vital::algo::estimate_fundamental_matrix_sptr f_estimator_;

  // Logger handle
  vital::logger_handle_t m_logger;
};


// ----------------------------------------------------------------------------
// Constructor
match_features_fundamental_matrix
::match_features_fundamental_matrix()
: d_(new priv)
{
  attach_logger( "arrows.core.match_features_fundamental_matrix" );
}


// Destructor
match_features_fundamental_matrix
::~match_features_fundamental_matrix()
{
}


// ----------------------------------------------------------------------------
// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
match_features_fundamental_matrix
::get_configuration() const
{
  vital::config_block_sptr config = algorithm::get_configuration();
  config->set_value("inlier_scale", d_->inlier_scale,
                    "The acceptable error distance (in pixels) between a measured point "
                    "and its epipolar line to be considered an inlier match.");
  config->set_value("min_required_inlier_count", d_->min_required_inlier_count,
                    "The minimum required inlier point count. If there are less "
                    "than this many inliers, no matches will be returned.");
  config->set_value("min_required_inlier_percent", d_->min_required_inlier_percent,
                    "The minimum required percentage of inlier points. If the "
                    "percentage of points considered inliers is less than this "
                    "amount, no matches will be returned.");
  config->set_value("motion_filter_percentile", d_->motion_filter_percentile,
                    "If less than 1.0, find this percentile of the motion "
                    "magnitude and filter matches with motion larger than "
                    "twice this value.  This helps remove outlier matches "
                    "when the motion between images is small.");

  // nested algorithm configurations
  vital::algo::estimate_fundamental_matrix::get_nested_algo_configuration("fundamental_matrix_estimator",
                                                                           config, d_->f_estimator_);
  vital::algo::match_features::get_nested_algo_configuration("feature_matcher", config,
                                                             d_->matcher_);

  return config;
}


// ----------------------------------------------------------------------------
void
match_features_fundamental_matrix
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Set nested algorithm configurations
  vital::algo::estimate_fundamental_matrix::set_nested_algo_configuration("fundamental_matrix_estimator",
                                                                           config, d_->f_estimator_);
  vital::algo::match_features::set_nested_algo_configuration("feature_matcher", config, d_->matcher_);

  // Other parameters
  d_->inlier_scale = config->get_value<double>("inlier_scale");
  d_->min_required_inlier_count = config->get_value<int>("min_required_inlier_count");
  d_->min_required_inlier_percent = config->get_value<double>("min_required_inlier_percent");
  d_->motion_filter_percentile =
    config->get_value<double>("motion_filter_percentile");
}


// ----------------------------------------------------------------------------
bool
match_features_fundamental_matrix
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;
  double motion_filter_percentile =
    config->get_value<double>("motion_filter_percentile",
                              d_->motion_filter_percentile);
  // this algorithm is optional
  if (motion_filter_percentile < 0.0 || motion_filter_percentile > 1.0)
  {
    config_valid = false;
  }

  return (
    vital::algo::estimate_fundamental_matrix::check_nested_algo_configuration("fundamental_matrix_estimator", config)
    &&
    vital::algo::match_features::check_nested_algo_configuration("feature_matcher", config)
    &&
    config_valid
  );
}


namespace {

// compute the p-th percentile of the data
double percentile(std::vector<double> const & data, double p)
{
  p = std::max(0.0, std::min(1.0, p));
  std::vector<double> d(data);
  size_t nth_idx = static_cast<size_t>(d.size() * p);
  std::nth_element(d.begin(), d.begin() + nth_idx, d.end());
  return d[nth_idx];
}
}

// ----------------------------------------------------------------------------
// Match one set of features and corresponding descriptors to another
match_set_sptr
match_features_fundamental_matrix
::match(feature_set_sptr feat1, descriptor_set_sptr desc1,
        feature_set_sptr feat2, descriptor_set_sptr desc2) const
{
  if( !d_->matcher_ || !d_->f_estimator_ )
  {
    return match_set_sptr();
  }

  // compute the initial matches
  match_set_sptr init_matches = d_->matcher_->match(feat1, desc1, feat2, desc2);

  // estimate a fundamental_matrix from the initial matches
  std::vector<bool> inliers;
  fundamental_matrix_sptr F = d_->f_estimator_->estimate(feat1, feat2, init_matches,
                                                     inliers, d_->inlier_scale);
  int inlier_count = static_cast<int>(std::count(inliers.begin(), inliers.end(), true));
  LOG_INFO(logger(), "inlier ratio: " << inlier_count << "/" << inliers.size());

  // verify matching criteria are met
  if( !inlier_count || inlier_count < d_->min_required_inlier_count ||
      static_cast<double>(inlier_count)/inliers.size() < d_->min_required_inlier_percent )
  {
    return match_set_sptr(new simple_match_set());
  }

  // return the subset of inlier matches
  std::vector<vital::match> m = init_matches->matches();
  std::vector<vital::match> inlier_m;
  for( unsigned int i=0; i<inliers.size(); ++i )
  {
    if( inliers[i] )
    {
      inlier_m.push_back(m[i]);
    }
  }

  if (d_->motion_filter_percentile >= 1.0)
  {
    return match_set_sptr(new simple_match_set(inlier_m));
  }

  // Further filter the matches by motion amount to remove outliers.
  // For relatively small motions there may be outliers that agree
  // with the epipolar geometry but have unusually large motion.
  // Discard matches with motion above the twice the Nth percentile.
  auto f1 = feat1->features();
  auto f2 = feat2->features();
  std::vector<double> dists;
  dists.reserve(inlier_m.size());
  for (auto const& m : inlier_m)
  {
    vector_2d dist = f1[m.first]->loc() - f2[m.second]->loc();
    dists.push_back(dist.norm());
  }
  double max_dist = 2.0 * percentile(dists, d_->motion_filter_percentile);
  std::vector<vital::match> filtered_m;
  for (unsigned i = 0; i < inlier_m.size(); ++i)
  {
    if (dists[i] < max_dist)
    {
      filtered_m.push_back(inlier_m[i]);
    }
  }

  LOG_DEBUG(logger(), "Filtered " << inlier_m.size() - filtered_m.size() <<
                      " matches with motion greater than " << max_dist);

  return match_set_sptr(new simple_match_set(filtered_m));
}


} // end namespace core

} // end namespace arrows
} // end namespace kwiver
