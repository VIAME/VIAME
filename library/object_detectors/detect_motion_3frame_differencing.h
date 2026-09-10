// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV motion detection 3-frame difference algorithm impl interface

#ifndef KWIVER_ARROWS_OCV_THREE_FRAME_DIFFERENCING_H_
#define KWIVER_ARROWS_OCV_THREE_FRAME_DIFFERENCING_H_

#include <memory>


#include <viame/algorithm_framework/algo/detect_motion.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/core_types/timestamp.h>
#include <viame/algorithm_framework/vital_config.h>

#include "viame_object_detectors_export.h"

namespace kwiver {

namespace arrows {

namespace ocv {

/// OCV implementation of detect_motion using three-frame differencing
class VIAME_OBJECT_DETECTORS_EXPORT detect_motion_3frame_differencing
  : public vital::algo::detect_motion
{
public:
  PLUGGABLE_IMPL(
    detect_motion_3frame_differencing,
    "OCV implementation of detect_motion using three-frame differencing",

    PARAM_DEFAULT(
      frame_separation,
      std::size_t,
      "Number of frames of separation for difference "
      "calculation. Queue of collected images must be twice this "
      "value before a three-frame difference can be "
      "calculated.",
      1 ),

    PARAM_DEFAULT(
      jitter_radius,
      int,
      "Radius of jitter displacement (pixels) expected in the "
      "image due to imperfect stabilization. The image "
      "differencing process will search for the lowest-magnitude "
      "difference in a neighborhood with radius equal to "
      "jitter_radius.",
      0 ),

    PARAM_DEFAULT(
      max_foreground_fract,
      double,
      "Specifies the maximum expected fraction of the scene "
      "that may contain foreground movers at any time. When the "
      "fraction of pixels determined to be in motion exceeds "
      "this value, the background model is assumed to be "
      "invalid (e.g., due to excessive camera motion) and is "
      "reset. The default value of 1 indicates that no checking "
      "is done.",
      1 ),

    PARAM_DEFAULT(
      max_foreground_fract_thresh,
      double,
      "To be used in conjunction with max_foreground_fract, this "
      "parameter defines the threshold for foreground in order "
      "to determine if the maximum fraction of foreground has "
      "been exceeded.",
      -1 ),

    PARAM_DEFAULT(
      debug_dir,
      std::string,
      "Output debug images to this directory.",
      "" )
  );

  /// Destructor
  virtual ~detect_motion_3frame_differencing() noexcept;

  /// Check that the algorithm's configuration vital::config_block is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Detect motion from a sequence of images
  ///
  /// This method detects motion of foreground objects within a
  /// sequence of images in which the background remains stationary.
  /// Sequential images are passed one at a time. Motion estimates
  /// are returned for each image as a heat map with higher values
  /// indicating greater confidence.
  ///
  /// \param ts Timestamp for the input image
  /// \param image Image from a sequence
  /// \param reset_model Indicates that the model should be reset, for example,
  /// due to changes in lighting condition or
  /// camera pose
  ///
  /// \returns A heat map image is returned indicating the confidence
  /// that motion occurred at each pixel. Heat map image is single channel
  /// and has the same width and height dimensions as the input image.
  kwiver::vital::image_container_sptr
  process_image(
    const kwiver::vital::timestamp& ts,
    const kwiver::vital::image_container_sptr image,
    bool reset_model ) override;

private:
  void initialize() override;
  void set_configuration_internal( vital::config_block_sptr config ) override;
  // private implementation class
  class priv;

  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
