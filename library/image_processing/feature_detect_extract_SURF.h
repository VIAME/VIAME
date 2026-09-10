// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV SURF feature detector and extractor wrapper

#ifndef KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SURF_H_
#define KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SURF_H_

#include <opencv2/opencv_modules.hpp>
#if defined( HAVE_OPENCV_NONFREE ) && defined( HAVE_OPENCV_XFEATURES2D )

#include <viame/image_processing/detect_features.h>
#include <viame/image_processing/extract_descriptors.h>
#include "viame_image_processing_export.h"

#include <string>

namespace kwiver {

namespace arrows {

namespace ocv {

class VIAME_IMAGE_PROCESSING_EXPORT detect_features_SURF
  : public ocv::detect_features
{
public:
  PLUGGABLE_IMPL(
    detect_features_SURF,
    "OpenCV feature detection via the SURF algorithm",

    PARAM_DEFAULT(
      hessian_threshold, double,
      "Threshold for hessian keypoint detector used in SURF",
      100.0 ),

    PARAM_DEFAULT(
      n_octaves, int,
      "Number of pyramid octaves the keypoint detector will "
      "use.", 4 ),

    PARAM_DEFAULT(
      n_octaves_layers, int,
      "Number of octave layers within each octave.",
      3 ),

    PARAM_DEFAULT(
      extended, bool,
      "Extended descriptor flag (true - use extended "
      "128-element descriptors; false - use 64-element "
      "descriptors).",
      false ),

    PARAM_DEFAULT(
      upright, bool,
      "Up-right or rotated features flag (true - do not "
      "compute orientation of features; false - "
      "compute orientation).",
      false )
  );

  /// Destructor
  virtual ~detect_features_SURF();

  /// Check that the algorithm's configuration vital::config_block is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  void update_detector_parameters()  const override;
};

class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors_SURF
  : public ocv::extract_descriptors
{
public:
  PLUGGABLE_IMPL(
    extract_descriptors_SURF,
    "OpenCV feature-point descriptor extraction via the SURF algorithm",

    PARAM_DEFAULT(
      hessian_threshold, double,
      "Threshold for hessian keypoint detector used in SURF",
      100 ),

    PARAM_DEFAULT(
      n_octaves, int,
      "Number of pyramid octaves the keypoint detector will "
      "use.", 4 ),

    PARAM_DEFAULT(
      n_octaves_layers, int,
      "Number of octave layers within each octave.",
      3 ),

    PARAM_DEFAULT(
      extended, bool,
      "Extended descriptor flag (true - use extended "
      "128-element descriptors; false - use 64-element "
      "descriptors).",
      false ),

    PARAM_DEFAULT(
      upright, bool,
      "Up-right or rotated features flag (true - do not "
      "compute orientation of features; false - "
      "compute orientation).",
      false )
  );

  /// Destructor
  virtual ~extract_descriptors_SURF();

  bool check_configuration( vital::config_block_sptr config ) const override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  void update_extractor_parameters()  const override;
};

#define KWIVER_OCV_HAS_SURF

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif

#endif
