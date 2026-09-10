// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for OCV refine detections write to disk algorithm

#ifndef KWIVER_ARROWS_OCV_REFINE_DETECTIONS_WRITE_TO_DISK_H_
#define KWIVER_ARROWS_OCV_REFINE_DETECTIONS_WRITE_TO_DISK_H_

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/refine_detections.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A class for drawing various information about feature tracks
class VIAME_IMAGE_PROCESSING_EXPORT refine_detections_write_to_disk
  : public vital::algo::refine_detections
{
public:
  // Registered as "ocv_write" in register_algorithms.cxx
  PLUGGABLE_IMPL(
    refine_detections_write_to_disk,
    "Debugging process for writing out detections",
    PARAM_DEFAULT(
      pattern, std::string,
      "The output pattern for writing images to disk. "
      "Parameters that may be included in the pattern are (in formatting order)"
      "the detection category (a string), the source image filename or frame "
      "number (a string), and four values for the chip coordinate: "
      "top left x, top left y, width, height (all integers). "
      "A possible full pattern would be '%s-%s-%d-%d-%d-%d.png'. "
      "The pattern must contain the correct file extension.",
      "%s-%s+%d_%d_%dx%d.png" ),
    PARAM_DEFAULT(
      unknown_label, std::string,
      "Label to use in the output pattern for detections which carry no "
      "classification.",
      "unknown" )
  );

  /// Destructor
  virtual ~refine_detections_write_to_disk();

  /// Check that the algorithm's currently configuration is valid
  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Refine all object detections on the provided image
  ///
  /// This method analyzes the supplied image and and detections on it,
  /// returning a refined set of detections.
  ///
  /// \param image_data the image pixels
  /// \param detections detected objects
  /// \returns vector of image objects refined
  vital::detected_object_set_sptr
  refine(
    vital::image_container_sptr image_data,
    vital::detected_object_set_sptr detections ) const override;

private:
  // Variables
  mutable unsigned detection_counter { 0 };
  mutable unsigned frame_counter { 0 };
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
