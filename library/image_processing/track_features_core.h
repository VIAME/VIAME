// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining the track_features_core algorithm

#ifndef ARROWS_PLUGINS_CORE_TRACK_FEATURES_CORE_H_
#define ARROWS_PLUGINS_CORE_TRACK_FEATURES_CORE_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/track_features.h>
#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/image_container.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

#include <viame/algorithm_framework/algo/close_loops.h>
#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include <viame/algorithm_framework/algo/match_features.h>

namespace kwiver {

namespace arrows {

namespace core {

/// A basic feature tracker
class VIAME_IMAGE_PROCESSING_EXPORT track_features_core
  : public vital::algo::track_features
{
public:
  PLUGGABLE_IMPL(
    track_features_core,
    "Track features from frame to frame"
    " using feature detection, matching, and loop closure.",
    PARAM_DEFAULT(
      features_dir, kwiver::vital::config_path_t,
      "Path to a directory in which to read or write the feature "
      "detection and description files.\n"
      "Using this directory requires a feature_io algorithm.",
      "" ),
    PARAM(
      feature_detector, vital::algo::detect_features_sptr,
      "feature_detector" ),
    PARAM(
      descriptor_extractor, vital::algo::extract_descriptors_sptr,
      "descriptor_extractor" ),
    PARAM(
      feature_io, vital::algo::feature_descriptor_io_sptr,
      "feature_io" ),
    PARAM(
      feature_matcher, vital::algo::match_features_sptr,
      "feature_matcher" ),
    PARAM(
      loop_closer, vital::algo::close_loops_sptr,
      "loop_closer" )
  )

  /// Destructor
  virtual ~track_features_core() noexcept;

  /// Check that the algorithm's currently configuration is valid
  ///
  /// This checks solely within the provided \c config_block and not against
  /// the current state of the instance. This isn't static for inheritence
  /// reasons.
  ///
  /// \param config  The config block to check configuration of.
  ///
  /// \returns true if the configuration check passed and false if it didn't.
  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Extend a previous set of feature tracks using the current frame
  ///
  /// \throws image_size_mismatch_exception
  ///    When the given non-zero mask image does not match the size of the
  ///    dimensions of the given image data.
  ///
  /// \param [in] prev_tracks the feature tracks from previous tracking steps
  /// \param [in] frame_number the frame number of the current frame
  /// \param [in] image_data the image pixels for the current frame
  /// \param [in] mask Optional mask image that uses positive values to denote
  ///                  regions of the input image to consider for feature
  ///                  tracking. An empty sptr indicates no mask (default
  ///                  value).
  /// \returns an updated set of feature tracks including the current frame
  vital::feature_track_set_sptr
  track(
    vital::feature_track_set_sptr prev_tracks,
    vital::frame_id_t frame_number,
    vital::image_container_sptr image_data,
    vital::image_container_sptr mask = {} ) const override;

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
