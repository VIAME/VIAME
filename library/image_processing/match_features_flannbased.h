// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV flann-based feature matcher wrapper

#ifndef KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H
#define KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H

#include <memory>
#include <vector>

#include <viame/image_processing/match_features.h>
#include "viame_image_processing_export.h"

namespace kwiver {

namespace arrows {

namespace ocv {

/// Feature matcher implementation using OpenCV's FLANN-based feature matcher
class VIAME_IMAGE_PROCESSING_EXPORT match_features_flannbased
  : public ocv::match_features
{
public:
  PLUGGABLE_IMPL(
    match_features_flannbased,
    "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors).",

    PARAM_DEFAULT(
      cross_check, bool,
      "If cross-check filtering should be performed.",
      true ),

    PARAM_DEFAULT(
      cross_check_k, int,
      "Number of neighbors to use when cross checking",
      1 ),

    PARAM_DEFAULT(
      binary_descriptors, bool,
      "if false assume float descriptors (use l2 kdtree). "
      "if true assume binary descriptors (use lsh).",
      false )
  );

  /// Destructor
  virtual ~match_features_flannbased();

  /// Check that the algorithm's configuration vital::config_block is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

protected:
  /// Perform matching based on the underlying OpenCV implementation
  void ocv_match(
    const cv::Mat& descriptors1,
    const cv::Mat& descriptors2,
    std::vector< cv::DMatch >& matches ) const override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  class priv;

  KWIVER_UNIQUE_PTR( priv, p_ );
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif // KWIVER_ARROWS_MATCH_FEATURES_FLANNBASED_H
