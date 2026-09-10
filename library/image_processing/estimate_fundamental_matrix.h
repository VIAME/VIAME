// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV estimate_fundamental_matrix algorithm impl interface

#ifndef KWIVER_ARROWS_OCV_ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define KWIVER_ARROWS_OCV_ESTIMATE_FUNDAMENTAL_MATRIX_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A class that using OpenCV to estimate a fundamental matrix from matching 2D
/// points
class VIAME_IMAGE_PROCESSING_EXPORT estimate_fundamental_matrix
  : public vital::algo::estimate_fundamental_matrix
{
public:
  // "ocv",
  PLUGGABLE_IMPL(
    estimate_fundamental_matrix,
    "Use OpenCV to estimate a fundimental matrix from feature matches.",

    PARAM_DEFAULT(
      confidence_threshold,
      double,
      "Confidence that estimated matrix is correct, range (0.0, 1.0]",
      0.99 )
  );

  /// Destructor
  virtual ~estimate_fundamental_matrix();

  /// Check that the algorithm's configuration config_block is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Estimate a fundamental matrix from corresponding points
  ///
  /// If estimation fails, a NULL-containing sptr is returned
  ///
  /// \param [in]  pts1 the vector or corresponding points from the source image
  /// \param [in]  pts2 the vector of corresponding points from the destination
  /// image
  /// \param [out] inliers for each point pair, the value is true if
  ///                      this pair is an inlier to the fundamental matrix
  /// estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  vital::fundamental_matrix_sptr
  estimate(
    const std::vector< vital::vector_2d >& pts1,
    const std::vector< vital::vector_2d >& pts2,
    std::vector< bool >& inliers,
    double inlier_scale = 3.0 ) const override;
  using vital::algo::estimate_fundamental_matrix::estimate;

private:
  void initialize() override;
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
