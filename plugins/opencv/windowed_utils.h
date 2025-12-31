/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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

#ifndef VIAME_OPENCV_WINDOWED_UTILS_H
#define VIAME_OPENCV_WINDOWED_UTILS_H


#include "viame_opencv_export.h"

#include <opencv2/core/core.hpp>

#include <vital/algo/image_object_detector.h>
#include <vital/util/enum_converter.h>

namespace viame {

namespace kv = kwiver::vital;

enum rescale_option {
  DISABLED = 0,
  MAINTAIN_AR,
  SCALE,
  CHIP,
  CHIP_AND_ORIGINAL,
  ORIGINAL_AND_RESIZED,
  ADAPTIVE
};

ENUM_CONVERTER( rescale_option_converter, rescale_option,
    { "disabled",             DISABLED },
    { "maintain_ar",          MAINTAIN_AR },
    { "scale",                SCALE },
    { "chip",                 CHIP },
    { "chip_and_original",    CHIP_AND_ORIGINAL },
    { "original_and_resized", ORIGINAL_AND_RESIZED },
    { "adaptive",             ADAPTIVE }
  )

struct window_settings
{
  window_settings();
  ~window_settings() {}

  kv::config_block_sptr config() const;
  void set_config( kv::config_block_sptr cfg );

  rescale_option mode;
  double scale;
  int chip_width;
  int chip_height;
  int chip_step_width;
  int chip_step_height;
  int chip_edge_filter;
  double chip_edge_max_prob;
  int chip_adaptive_thresh;
  int batch_size;
  int min_detection_dim;
  bool original_to_chip_size;
  bool black_pad;
};

struct windowed_region_prop
{
  explicit windowed_region_prop( cv::Rect r, double s1 );

  explicit windowed_region_prop( cv::Rect r, int ef, bool rb,
    bool bb, double s1, int sx, int sy, double s2 );

  cv::Rect original_roi;
  int edge_filter;
  bool right_border;
  bool bottom_border;
  double scale1;
  int shiftx, shifty;
  double scale2;
};

double
scale_image_maintaining_ar(
  const cv::Mat& src,
  cv::Mat& dst,
  int width,
  int height,
  bool pad = false );

double
format_image(
  const cv::Mat& src,
  cv::Mat& dst,
  rescale_option option,
  double scale_factor,
  int width,
  int height,
  bool pad = false );

kv::detected_object_set_sptr
rescale_detections(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  double chip_edge_max_prob );

void
prepare_image_regions(
  const cv::Mat& image,
  const window_settings& settings,
  std::vector< cv::Mat >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties );

void scale_detections(
  kv::detected_object_set_sptr& detections,
  const windowed_region_prop& region_info );

kv::detected_object_set_sptr
scale_detections_to_region(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info );

void
scale_detections_to_region_with_mapping(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  std::vector< kv::detected_object_sptr >& original_detections,
  std::vector< kv::detected_object_sptr >& scaled_detections );

void
separate_boundary_detections(
  const kv::detected_object_set_sptr detections,
  int region_width,
  int region_height,
  kv::detected_object_set_sptr& boundary_detections,
  kv::detected_object_set_sptr& interior_detections );

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_UTILS_H */
