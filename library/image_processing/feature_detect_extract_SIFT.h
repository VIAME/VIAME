// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV SIFT feature detector and extractor wrapper

#ifndef KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SIFT_H_
#define KWIVER_ARROWS_FEATURE_DETECT_EXTRACT_SIFT_H_

#include <opencv2/opencv_modules.hpp>
#if defined( HAVE_OPENCV_NONFREE ) || defined( HAVE_OPENCV_XFEATURES2D )

#include <viame/image_processing/detect_features.h>
#include <viame/image_processing/extract_descriptors.h>
#include "viame_image_processing_export.h"

#include <string>

namespace kwiver {

namespace arrows {

namespace ocv {

class VIAME_IMAGE_PROCESSING_EXPORT detect_features_SIFT
  : public ocv::detect_features
{
public:
  PLUGGABLE_IMPL(
    detect_features_SIFT,
    "OpenCV feature detection via the SIFT algorithm",
    PARAM_DEFAULT(
      n_features, int,
      "The number of best features to retain. The features "
      "are ranked by their scores (measured in SIFT algorithm "
      "as the local contrast", 0 ),
    PARAM_DEFAULT(
      n_octave_layers, int,
      "The number of layers in each octave. 3 is the value "
      "used in D. Lowe paper. The number of octaves is "
      "computed automatically from the image resolution.",
      3 ),
    PARAM_DEFAULT(
      contrast_threshold, double,
      "The contrast threshold used to filter out weak "
      "features in semi-uniform (low-contrast) regions. The "
      "larger the threshold, the less features are produced "
      "by the detector.",
      0.04 ),
    PARAM_DEFAULT(
      edge_threshold, int,
      "The threshold used to filter out edge-like features. "
      "Note that the its meaning is different from the "
      "contrast_threshold, i.e. the larger the "
      "edge_threshold, the less features are filtered out "
      "(more features are retained).", 10 ),
    PARAM_DEFAULT(
      sigma, double,
      "The sigma of the Gaussian applied to the input image "
      "at the octave #0. If your image is captured with a "
      "weak camera with soft lenses, you might want to reduce "
      "the number.", 1.6 )
  );

  /// Destructor
  virtual ~detect_features_SIFT();

  bool check_configuration( vital::config_block_sptr config ) const override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  void update_detector_parameters()  const override;
};

class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors_SIFT
  : public ocv::extract_descriptors
{
public:
  PLUGGABLE_IMPL(
    extract_descriptors_SIFT,
    "OpenCV feature detection via the SIFT algorithm",
    PARAM_DEFAULT(
      n_features, int,
      "The number of best features to retain. The features "
      "are ranked by their scores (measured in SIFT algorithm "
      "as the local contrast", 0 ),
    PARAM_DEFAULT(
      n_octave_layers, int,
      "The number of layers in each octave. 3 is the value "
      "used in D. Lowe paper. The number of octaves is "
      "computed automatically from the image resolution.",
      3 ),
    PARAM_DEFAULT(
      contrast_threshold, double,
      "The contrast threshold used to filter out weak "
      "features in semi-uniform (low-contrast) regions. The "
      "larger the threshold, the less features are produced "
      "by the detector.",
      0.04 ),
    PARAM_DEFAULT(
      edge_threshold, int,
      "The threshold used to filter out edge-like features. "
      "Note that the its meaning is different from the "
      "contrast_threshold, i.e. the larger the "
      "edge_threshold, the less features are filtered out "
      "(more features are retained).", 10 ),
    PARAM_DEFAULT(
      sigma, double,
      "The sigma of the Gaussian applied to the input image "
      "at the octave #0. If your image is captured with a "
      "weak camera with soft lenses, you might want to reduce "
      "the number.", 1.6 )
  );

  /// Destructor
  virtual ~extract_descriptors_SIFT();

  bool check_configuration( vital::config_block_sptr config ) const override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  void update_extractor_parameters()  const override;
};

#define KWIVER_OCV_HAS_SIFT

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif // defined(HAVE_OPENCV_NONFREE) || defined(HAVE_OPENCV_XFEATURES2D)

#endif
