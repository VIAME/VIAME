// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_INITIALIZE_OBJECT_TRACKS_THRESHOLD_H_
#define KWIVER_ARROWS_INITIALIZE_OBJECT_TRACKS_THRESHOLD_H_

#include "viame_object_trackers_export.h"
#include <viame/algorithm_framework/viame_compiler_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/initialize_object_tracks.h>

#include <viame/algorithm_framework/algo/detected_object_filter.h>

namespace viame {

namespace core {

/// Initialize object tracks via simple single frame thresholding
class VIAME_OBJECT_TRACKERS_EXPORT initialize_object_tracks_threshold
  : public viame::algo::initialize_object_tracks
{
public:
  PLUGGABLE_IMPL(
    initialize_object_tracks_threshold,
    "Perform thresholding on detection confidence values to create tracks.",
    PARAM_DEFAULT(
      max_new_tracks, size_t,
      "Maximum number of new tracks to initialize on a single frame.",
      10000 ),
    PARAM(
      filter, viame::algo::detected_object_filter_sptr,
      "filter" )
  )

  /// Destructor
  virtual ~initialize_object_tracks_threshold() noexcept;

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

  /// Initialize new object tracks given detections.
  ///
  /// \param ts frame ID
  /// \param image contains the input image for the current frame
  /// \param detections detected object sets from the current frame
  /// \returns newly initialized tracks
  virtual viame::object_track_set_sptr
  initialize(
    viame::timestamp ts,
    viame::image_container_sptr image,
    viame::detected_object_set_sptr detections ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // namespace core

} // namespace viame

#endif
