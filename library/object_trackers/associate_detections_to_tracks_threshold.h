// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_ASSOCIATE_DETECTIONS_TO_TRACKS_THRESHOLD_H_
#define KWIVER_ARROWS_ASSOCIATE_DETECTIONS_TO_TRACKS_THRESHOLD_H_

#include "viame_object_trackers_export.h"
#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/associate_detections_to_tracks.h>

namespace viame {

namespace core {

/// Initialize object tracks via simple single frame thresholding
class VIAME_OBJECT_TRACKERS_EXPORT associate_detections_to_tracks_threshold
  : public viame::algo::associate_detections_to_tracks
{
public:
  PLUGGABLE_IMPL(
    associate_detections_to_tracks_threshold,
    "Associate detections to tracks via simple thresholding on the input matrix.",
    PARAM_DEFAULT(
      threshold, double,
      "Threshold to apply on the matrix.",
      0.50 ),
    PARAM_DEFAULT(
      higher_is_better, bool,
      "Whether values above or below the threshold indicate a better fit.",
      true )
  )

  /// Destructor
  virtual ~associate_detections_to_tracks_threshold() noexcept;

  /// Check that the algorithm's currently configuration is valid
  ///
  /// This checks solely within the provided \c config_block and not against
  /// the current state of the instance. This isn't static for inheritence
  /// reasons.
  ///
  /// \param config  The config block to check configuration of.
  ///
  /// \returns true if the configuration check passed and false if it didn't.
  virtual bool check_configuration( viame::config_block_sptr config ) const;

  /// Use cost matrices to assign detections to existing tracks
  ///
  /// \param ts frame ID
  /// \param image contains the input image for the current frame
  /// \param tracks active track set from the last frame
  /// \param detections detected object sets from the current frame
  /// \param matrix matrix containing detection to track association scores
  /// \param output the output updated detection set
  /// \param unused output detection set for any detections not associated
  /// \returns whether or not any tracks were updated
  virtual bool
  associate(
    viame::timestamp ts,
    viame::image_container_sptr image,
    viame::object_track_set_sptr tracks,
    viame::detected_object_set_sptr detections,
    viame::matrix_d matrix,
    viame::object_track_set_sptr& output,
    viame::detected_object_set_sptr& unused ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // namespace core

} // namespace viame

#endif
