/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_SIFT`, over the ported algorithm in sift.h
///
/// The config keys, their defaults and their descriptions are the ones the
/// python wrapper registered and `registry.json` records, so the shipped
/// configs keep resolving and keep meaning the same thing.

#ifndef VIAME_IMAGE_PROCESSING_SIFT_FEATURES_H
#define VIAME_IMAGE_PROCESSING_SIFT_FEATURES_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>
#include <viame/core_types/feature_set.h>

#include <vector>

namespace viame {

namespace kv = viame;

/// A feature set that also remembers each keypoint's packed octave.
///
/// SIFT needs it and `vital::feature` has nowhere to put it. The octave says
/// which level of the Gaussian pyramid a keypoint was found at, and therefore
/// which level its descriptor has to be sampled from; without it every
/// keypoint is described at the base of the pyramid, which is scale invariant
/// in name only.
///
/// This is what the python implementation's `OCVFeatureSet` was for, and it
/// behaves the same way: `extract` recovers the octaves when the set it is
/// handed is one of these, and uses zero for every keypoint when it is not.
/// A set from another detector therefore gets described at the base level,
/// which is what cv2 does with the same input.
class VIAME_IMAGE_PROCESSING_EXPORT sift_feature_set
  : public kv::simple_feature_set
{
public:
  sift_feature_set( std::vector< kv::feature_sptr > const& features,
                    std::vector< int > const& octaves )
    : kv::simple_feature_set( features ), octaves_( octaves ) {}

  std::vector< int > const& octaves() const { return octaves_; }

private:
  std::vector< int > octaves_;
};

#define VIAME_SIFT_PARAMS \
    PARAM_DEFAULT( \
      n_features, int, \
      "The number of best features to retain. The features are ranked by " \
      "their scores (measured in SIFT algorithm as the local contrast).", \
      0 ), \
    PARAM_DEFAULT( \
      n_octave_layers, int, \
      "The number of layers in each octave. 3 is the value used in D. Lowe " \
      "paper. The number of octaves is computed automatically from the " \
      "image resolution.", \
      3 ), \
    PARAM_DEFAULT( \
      contrast_threshold, double, \
      "The contrast threshold used to filter out weak features in " \
      "semi-uniform (low-contrast) regions. The larger the threshold, the " \
      "less features are produced by the detector.", \
      0.04 ), \
    PARAM_DEFAULT( \
      edge_threshold, int, \
      "The threshold used to filter out edge-like features. Note that the " \
      "its meaning is different from the contrast_threshold, i.e. the larger " \
      "the edge_threshold, the less features are filtered out (more " \
      "features are retained).", \
      10 ), \
    PARAM_DEFAULT( \
      sigma, double, \
      "The sigma of the Gaussian applied to the input image at the octave " \
      "#0. If your image is captured with a weak camera with soft lenses, " \
      "you might want to reduce the number.", \
      1.6 )

/// @brief Detect SIFT keypoints.
class VIAME_IMAGE_PROCESSING_EXPORT detect_features_SIFT
  : public kv::algo::detect_features
{
public:
  PLUGGABLE_IMPL_NAMED(
    detect_features_SIFT,
    "ocv_SIFT",
    "OpenCV feature detection via the SIFT algorithm",
    VIAME_SIFT_PARAMS )

  virtual ~detect_features_SIFT();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::feature_set_sptr detect(
    kv::image_container_sptr image_data,
    kv::image_container_sptr mask ) const override;

private:
  void initialize() override;
};

/// @brief Describe features with SIFT.
class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors_SIFT
  : public kv::algo::extract_descriptors
{
public:
  PLUGGABLE_IMPL_NAMED(
    extract_descriptors_SIFT,
    "ocv_SIFT",
    "OpenCV feature detection via the SIFT algorithm",
    VIAME_SIFT_PARAMS )

  virtual ~extract_descriptors_SIFT();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::descriptor_set_sptr extract(
    kv::image_container_sptr image_data,
    kv::feature_set_sptr& features,
    kv::image_container_sptr image_mask ) const override;

private:
  void initialize() override;
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_SIFT_FEATURES_H
