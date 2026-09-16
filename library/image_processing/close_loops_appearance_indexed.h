// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Appearance indexed close loops algorithm impl interface

#ifndef KWIVER_CORE_CLOSE_LOOPS_APPEARANCE_INDEXED_H_
#define KWIVER_CORE_CLOSE_LOOPS_APPEARANCE_INDEXED_H_

#include <string>

#include <viame/algorithm_framework/algo/close_loops.h>
#include <viame/algorithm_framework/viame_compiler_config.h>

#include <viame/algorithm_framework/algo/algorithm.txx>

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>
#include <viame/algorithm_framework/algo/match_descriptor_sets.h>
#include <viame/algorithm_framework/algo/match_features.h>

namespace viame {

namespace core {

/// Loop closure algorithm that using appearance indexing for fast matching
class VIAME_IMAGE_PROCESSING_EXPORT close_loops_appearance_indexed
  : public viame::algo::close_loops
{
public:
  PLUGGABLE_IMPL(
    close_loops_appearance_indexed,
    "Uses bag of words index to close loops.",
    PARAM_DEFAULT(
      min_loop_inlier_matches, size_t,
      "The minimum number of inlier feature matches to accept a loop "
      "connection and join tracks",
      128 ),
    PARAM_DEFAULT(
      geometric_verification_inlier_threshold, double,
      "inlier threshold for fundamental matrix based geometric verification "
      "of loop closure in pixels",
      2.0 ),
    PARAM_DEFAULT(
      max_loop_attempts_per_frame, int,
      "The maximum number of loop closure attempts to make per frame",
      200 ),
    PARAM_DEFAULT(
      tracks_in_common_to_skip_loop_closing, int,
      "If this or more tracks are in common between two frames then don't try "
      "to complete a loop with them",
      0 ),
    PARAM_DEFAULT(
      skip_loop_detection_track_i_over_u_threshold, float,
      "skip loop detection if intersection over union of track ids in two "
      "frames is greater than this",
      0.5f ),
    PARAM_DEFAULT(
      min_loop_inlier_fraction, float,
      "Inlier fraction must be this high to accept a loop completion",
      0.5f ),
    PARAM(
      match_features, viame::algo::match_features_sptr,
      "match_features" ),
    PARAM(
      bag_of_words_matching, viame::algo::match_descriptor_sets_sptr,
      "bag_of_words_matching" ),
    PARAM(
      fundamental_mat_estimator, viame::algo::estimate_fundamental_matrix_sptr,
      "fundamental_mat_estimator" )
  )

  /// Destructor
  virtual ~close_loops_appearance_indexed();

  /// Find loops in a feature_track_set

  /// Attempt to perform closure operation and stitch tracks together.
  ///
  /// \param frame_number the frame number of the current frame
  /// \param input the input feature track set to stitch
  /// \param image image data for the current frame
  /// \param mask Optional mask image where positive values indicate
  ///                 regions to consider in the input image.
  /// \returns an updated set of feature tracks after the stitching operation
  virtual viame::feature_track_set_sptr
  stitch(
    viame::frame_id_t frame_number,
    viame::feature_track_set_sptr input,
    viame::image_container_sptr image,
    viame::image_container_sptr mask = viame::
    image_container_sptr() ) const;

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

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // namespace core

} // namespace viame

#endif
