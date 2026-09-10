// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining the close_loops_bad_frames_only algorithm

#ifndef KWIVER_ARROWS_CORE_CLOSE_LOOPS_BAD_FRAMES_ONLY_H_
#define KWIVER_ARROWS_CORE_CLOSE_LOOPS_BAD_FRAMES_ONLY_H_

#include "viame_image_processing_export.h"

#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/image_container.h>

#include <viame/algorithm_framework/algo/close_loops.h>
#include <viame/algorithm_framework/algo/match_features.h>
#include <viame/algorithm_framework/config/config_block.h>

#include <viame/algorithm_framework/algo/algorithm.txx>

namespace kwiver {

namespace arrows {

namespace core {

/// Attempts to stitch over incomplete or bad input frames.
///
/// This class attempts to only make short term loop closures
/// due to bad or incomplete feature point tracking. It operates on the
/// principle that when a bad frame occurs, there is generally a lower
/// percentage of feature tracks.
class VIAME_IMAGE_PROCESSING_EXPORT close_loops_bad_frames_only
  : public vital::algo::close_loops
{
public:
  PLUGGABLE_IMPL(
    close_loops_bad_frames_only,
    "Attempts short-term loop closure based on percentage "
    "of feature points tracked.",
    PARAM_DEFAULT(
      enabled, bool,
      "Should bad frame detection be enabled? This option will attempt to "
      "bridge the gap between frames which don't meet certain criteria "
      "(percentage of feature points tracked) and will instead attempt "
      "to match features on the current frame against past frames to "
      "meet this criteria. This is useful when there can be bad frames.",
      true ),
    PARAM_DEFAULT(
      percent_match_req, double,
      "The required percentage of features needed to be matched for a "
      "stitch to be considered successful (value must be between 0.0 and "
      "1.0).",
      0.35 ),
    PARAM_DEFAULT(
      new_shot_length, size_t,
      "Number of frames for a new shot to be considered valid before "
      "attempting to stitch to prior shots.",
      2 ),
    PARAM_DEFAULT(
      max_search_length, size_t,
      "Maximum number of frames to search in the past for matching to "
      "the end of the last shot.",
      5 ),
    PARAM(
      feature_matcher, kwiver::vital::algo::match_features_sptr,
      "feature_matcher" )
  )
  /// Destructor
  virtual ~close_loops_bad_frames_only() = default;

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

  /// Perform basic shot stitching for bad frame detection
  ///
  /// \param [in] frame_number the frame number of the current frame
  /// \param [in] input the input feature track set to stitch
  /// \param [in] image image data for the current frame
  /// \param [in] mask Optional mask image where positive values indicate
  ///                  regions to consider in the input image.
  /// \returns an updated set a feature tracks after the stitching operation
  virtual vital::feature_track_set_sptr
  stitch(
    vital::frame_id_t frame_number,
    vital::feature_track_set_sptr input,
    vital::image_container_sptr image,
    vital::image_container_sptr mask = vital::image_container_sptr() ) const;

protected:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
};

} // end namespace algo

} // end namespace arrows

} // end namespace kwiver

#endif
