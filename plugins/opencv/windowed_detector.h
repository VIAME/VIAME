/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_WINDOWED_DETECTOR_H
#define VIAME_OPENCV_WINDOWED_DETECTOR_H


#include "viame_opencv_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/algo/algorithm.txx>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary other detector over an image
 *
 * This process should be moved to core from ocv when able
 */
class VIAME_OPENCV_EXPORT windowed_detector
  : public kwiver::vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    windowed_detector,
    "Window some other arbitrary detector across the image",
    PARAM_DEFAULT(
      mode, std::string,
      "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
      "chip, chip_and_original, original_and_resized, or adaptive.",
      "disabled" ),
    PARAM_DEFAULT(
      scale, double,
      "Image scaling factor used when mode is scale or chip.",
      1.0 ),
    PARAM_DEFAULT(
      chip_width, int,
      "When in chip mode, the chip width.",
      1000 ),
    PARAM_DEFAULT(
      chip_height, int,
      "When in chip mode, the chip height.",
      1000 ),
    PARAM_DEFAULT(
      chip_step_width, int,
      "When in chip mode, the chip step size between chips.",
      500 ),
    PARAM_DEFAULT(
      chip_step_height, int,
      "When in chip mode, the chip step size between chips.",
      500 ),
    PARAM_DEFAULT(
      chip_edge_filter, int,
      "If using chipping, filter out detections this pixel count near borders.",
      -1 ),
    PARAM_DEFAULT(
      chip_edge_max_prob, double,
      "If using chipping, maximum type probability for edge detections",
      -1.0 ),
    PARAM_DEFAULT(
      chip_adaptive_thresh, int,
      "If using adaptive selection, total pixel count at which we start to chip.",
      2000000 ),
    PARAM_DEFAULT(
      batch_size, int,
      "Optional processing batch size to send to the detector.",
      1 ),
    PARAM_DEFAULT(
      min_detection_dim, int,
      "Minimum detection dimension in original image space.",
      1 ),
    PARAM_DEFAULT(
      original_to_chip_size, bool,
      "Optionally enforce the input image is the specified chip size",
      false ),
    PARAM_DEFAULT(
      black_pad, bool,
      "Black pad the edges of resized chips to ensure consistent dimensions",
      false ),
    PARAM(
      detector, kwiver::vital::algo::image_object_detector_sptr,
      "Algorithm pointer to nested detector" )
  )

  virtual ~windowed_detector() = default;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

private:
};

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_DETECTOR_H */
