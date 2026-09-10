// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining the core match_features_homography algorithm

#ifndef KWIVER_ARROWS_CORE_MATCH_FEATURES_HOMOGRAPHY_H_
#define KWIVER_ARROWS_CORE_MATCH_FEATURES_HOMOGRAPHY_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/filter_features.h>

#include <viame/algorithm_framework/algo/algorithm.txx>

#include <viame/algorithm_framework/algo/estimate_homography.h>
#include <viame/algorithm_framework/algo/match_features.h>
#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

namespace arrows {

namespace core {

/// Combines a feature matchers, homography estimation, and filtering
///
///  This is a meta-algorithm for feature matching that combines one or more
///  other feature matchers with homography estimation and feature filtering.
///  The algorithm applies another configurable feature matcher algorithm and
///  then applies a homography estimation algorithm to the resulting matches.
///  Outliers to the fit homography are discarded from the set of matches.
///
///  If a second matcher algorithm is provided, this algorithm will warp the
///  feature locations by the estimated homography before applying the second
///  matching algorithm to the aligned points.  This approach is useful for
///  finding weak matches that were missed by the first matcher but are
///  easier to detect once approximate location is known.  A good choice for
///  the second matcher is vxl::match_features_constrained.
///
///  If a filter_features algorithm is provided, this will be run on the
///  input features \b before running the first matcher.  The second matcher
///  will then run on the \b original unfilter features.  This allows, for
///  example, a slower but more robust feature matcher to run on a subset
///  of the strongest feature points in order to quickly establish an
///  and estimated homography.  Then a second, fast matcher can pick up
///  the additional weak matches using the constraint that the location
///  in the image is now known approximately.
class VIAME_IMAGE_PROCESSING_EXPORT match_features_homography
  : public vital::algo::match_features
{
public:
  PLUGGABLE_IMPL(
    match_features_homography,
    "Use an estimated homography as a geometric filter"
    " to remove outlier matches.",
    PARAM_DEFAULT(
      inlier_scale, double,
      "The acceptable error distance (in pixels) between warped "
      "and measured points to be considered an inlier match. "
      "Note that this scale is multiplied by the average scale of "
      "the features being matched at each stage.",
      1.0 ),
    PARAM_DEFAULT(
      min_required_inlier_count, int,
      "The minimum required inlier point count. If there are less "
      "than this many inliers, no matches will be output.",
      0 ),
    PARAM_DEFAULT(
      min_required_inlier_percent, double,
      "The minimum required percentage of inlier points. If the "
      "percentage of points considered inliers is less than this "
      "amount, no matches will be output.",
      0.0 ),
    PARAM(
      homography_estimator,
      vital::algo::estimate_homography_sptr,
      "homography_estimator" ),
    PARAM(
      feature_matcher1,
      vital::algo::match_features_sptr,
      "feature_matcher1" ),
    PARAM(
      feature_matcher2,
      vital::algo::match_features_sptr,
      "feature_matcher2" ),
    PARAM(
      filter_features,
      vital::algo::filter_features_sptr,
      "filter_features" )
  )

  /// Destructor
  virtual ~match_features_homography();

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Match one set of features and corresponding descriptors to another
  ///
  /// \param [in] feat1 the first set of features to match
  /// \param [in] desc1 the descriptors corresponding to \a feat1
  /// \param [in] feat2 the second set of features to match
  /// \param [in] desc2 the descriptors corresponding to \a feat2
  /// \returns a set of matching indices from \a feat1 to \a feat2
  virtual vital::match_set_sptr
  match(
    vital::feature_set_sptr feat1, vital::descriptor_set_sptr desc1,
    vital::feature_set_sptr feat2, vital::descriptor_set_sptr desc2 ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace algo

} // end namespace arrows

} // end namespace kwiver

#endif
