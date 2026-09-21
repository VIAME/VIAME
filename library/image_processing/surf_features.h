/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_SURF`, over the ported algorithm in surf.h
///
/// The config keys, their defaults and their descriptions are the ones the
/// python wrapper registered and `registry.json` records, so the shipped
/// configs -- which are identical to `main`'s -- keep resolving and keep
/// meaning the same thing. Note `n_octaves_layers`, with the extra s: that is
/// the key that was registered, and it stays.

#ifndef VIAME_IMAGE_PROCESSING_SURF_FEATURES_H
#define VIAME_IMAGE_PROCESSING_SURF_FEATURES_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

namespace kv = viame;

#define VIAME_SURF_PARAMS \
    PARAM_DEFAULT( \
      hessian_threshold, double, \
      "Threshold for hessian keypoint detector used in SURF", \
      100.0 ), \
    PARAM_DEFAULT( \
      n_octaves, int, \
      "Number of pyramid octaves the keypoint detector will use.", \
      4 ), \
    PARAM_DEFAULT( \
      n_octaves_layers, int, \
      "Number of octave layers within each octave.", \
      3 ), \
    PARAM_DEFAULT( \
      extended, bool, \
      "Extended descriptor flag (true - use extended 128-element " \
      "descriptors; false - use 64-element descriptors).", \
      false ), \
    PARAM_DEFAULT( \
      upright, bool, \
      "Up-right or rotated features flag (true - do not compute orientation " \
      "of features; false - compute orientation).", \
      false )

/// @brief Detect SURF keypoints.
class VIAME_IMAGE_PROCESSING_EXPORT detect_features_SURF
  : public kv::algo::detect_features
{
public:
  PLUGGABLE_IMPL_NAMED(
    detect_features_SURF,
    "ocv_SURF",
    "OpenCV feature detection via the SURF algorithm",
    VIAME_SURF_PARAMS )

  virtual ~detect_features_SURF();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::feature_set_sptr detect(
    kv::image_container_sptr image_data,
    kv::image_container_sptr mask ) const override;

private:
  void initialize() override;
};

/// @brief Describe features with SURF.
class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors_SURF
  : public kv::algo::extract_descriptors
{
public:
  PLUGGABLE_IMPL_NAMED(
    extract_descriptors_SURF,
    "ocv_SURF",
    "OpenCV feature detection via the SURF algorithm",
    VIAME_SURF_PARAMS )

  virtual ~extract_descriptors_SURF();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::descriptor_set_sptr extract(
    kv::image_container_sptr image_data,
    kv::feature_set_sptr& features,
    kv::image_container_sptr image_mask ) const override;

private:
  void initialize() override;
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_SURF_FEATURES_H
