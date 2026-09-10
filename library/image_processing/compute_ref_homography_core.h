// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining the compute_ref_homography algorithm

#ifndef KWIVER_ARROWS_CORE_COMPUTE_REF_HOMOGRAPHY_CORE_H_
#define KWIVER_ARROWS_CORE_COMPUTE_REF_HOMOGRAPHY_CORE_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

#include <viame/algorithm_framework/algo/compute_ref_homography.h>
#include <viame/algorithm_framework/algo/estimate_homography.h>
#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/homography.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace arrows {

namespace core {

/// Default impl class for mapping each image to some reference image.
///
/// This class differs from estimate_homographies in that estimate_homographies
/// simply performs a homography regression from matching feature points. This
/// class is designed to generate different types of homographies from input
/// feature tracks, which can transform each image back to the same coordinate
/// space derived from some initial refrerence image.
///
/// This implementation is state-based and is meant to be run in an online
/// fashion, i.e. run against a track set that has been iteratively updated on
/// successive non-regressing frames. This is ideal for when it is desired to
/// compute reference frames on all frames in a sequence.
class VIAME_IMAGE_PROCESSING_EXPORT compute_ref_homography_core
  : public vital::algo::compute_ref_homography
{
public:
  PLUGGABLE_IMPL(
    compute_ref_homography_core,
    "Default online sequential-frame reference homography estimator.",
    PARAM_DEFAULT(
      use_backproject_error,
      bool,
      "Should we remove extra points if the backproject error is high?",
      false ),
    PARAM_DEFAULT(
      backproject_threshold_sqr,
      double,
      "Backprojection threshold in terms of L2 distance squared "
      "(number of pixels)",
      16.0 ),
    PARAM_DEFAULT(
      forget_track_threshold,
      size_t,
      "After how many frames should we forget all info about a track?",
      5 ),
    PARAM_DEFAULT(
      min_track_length,
      size_t,
      "Minimum track length to use for homography regression",
      1 ),
    PARAM_DEFAULT(
      inlier_scale,
      double,
      "The acceptable error distance (in pixels) between warped "
      "and measured points to be considered an inlier match.",
      2.0 ),
    PARAM_DEFAULT(
      minimum_inliers, size_t,
      "Minimum number of matches required between source and "
      "reference planes for valid homography estimation.",
      4 ),
    PARAM_DEFAULT(
      allow_ref_frame_regression, bool,
      "Allow for the possibility of a frame, N, to have a "
      "reference frame, A, when a frame M < N has a reference frame B > A "
      "(assuming frames were sequentially iterated over with this algorithm).",
      true ),
    PARAM(
      estimator, vital::algo::estimate_homography_sptr,
      "Homography estimator"
    )
  )

  /// Default Destructor
  virtual ~compute_ref_homography_core();

  /// Check that the algorithm's currently configuration is valid
  ///
  /// This checks solely within the provided \c config_block and not against
  /// the current state of the instance. This isn't static for inheritence
  /// reasons.
  ///
  /// \param config  The config block to check configuration of.
  ///
  /// \returns true if the configuration check passed and false if it didn't.
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Estimate the transformation which maps some frame to a reference frame
  ///
  /// Similarly to track_features, this class was designed to be called in
  /// an online fashion for each sequential frame.
  ///
  /// \param frame_number frame identifier for the current frame
  /// \param tracks the set of all tracked features from the image
  /// \return estimated homography
  virtual vital::f2f_homography_sptr
  estimate(
    vital::frame_id_t frame_number,
    vital::feature_track_set_sptr tracks ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif
