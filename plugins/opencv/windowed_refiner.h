/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef VIAME_OPENCV_WINDOWED_REFINER_H
#define VIAME_OPENCV_WINDOWED_REFINER_H


#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/algo/algorithm.txx>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary detection refiner over an image
 *
 * This process applies a detection refinement algorithm across multiple
 * windowed regions of an image, scaling input detections to each region.
 */
class VIAME_OPENCV_EXPORT windowed_refiner
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL(
    windowed_refiner,
    "Window some other arbitrary refiner across the image",
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
    PARAM_DEFAULT(
      process_boundary_dets, bool,
      "Pass through detections touching tile boundaries unmodified in refiner",
      false ),
    PARAM_DEFAULT(
      overlapping_proc_once, bool,
      "Only refine each detection once if it appears in multiple tiles",
      true ),
    PARAM(
      refiner, kwiver::vital::algo::refine_detections_sptr,
      "Algorithm pointer to nested refiner" )
  )

  virtual ~windowed_refiner();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr detections ) const;

private:


};

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_REFINER_H */
