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

#ifndef VIAME_CORE_WINDOWED_UTILS_H
#define VIAME_CORE_WINDOWED_UTILS_H

#include "viame_core_export.h"

#include <vital/types/image_container.h>
#include <vital/algo/image_object_detector.h>
#include <vital/util/enum_converter.h>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Rescale option enum - shared between core and opencv versions
// =============================================================================
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

// =============================================================================
// Simple rectangle struct to replace cv::Rect
// =============================================================================
struct VIAME_CORE_EXPORT image_rect
{
  int x, y, width, height;

  image_rect() : x( 0 ), y( 0 ), width( 0 ), height( 0 ) {}
  image_rect( int x_, int y_, int w_, int h_ )
    : x( x_ ), y( y_ ), width( w_ ), height( h_ ) {}
};

// =============================================================================
// Window settings configuration
// =============================================================================
struct VIAME_CORE_EXPORT window_settings
{
  window_settings();
  ~window_settings() {}

  /// Get full configuration (for detector/refiner)
  kv::config_block_sptr config() const;
  void set_config( kv::config_block_sptr cfg );

  /// Get chip-only configuration (for trainer - excludes detector-specific settings)
  /// This includes: mode, scale, chip_width, chip_height, chip_step_width,
  /// chip_step_height, chip_adaptive_thresh, original_to_chip_size, black_pad
  kv::config_block_sptr chip_config() const;
  void set_chip_config( kv::config_block_sptr cfg );

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

// =============================================================================
// Region properties for windowed processing
// =============================================================================
struct VIAME_CORE_EXPORT windowed_region_prop
{
  explicit windowed_region_prop( image_rect r, double s1 );

  explicit windowed_region_prop( image_rect r, int ef, bool rb,
    bool bb, double s1, int sx, int sy, double s2 );

  image_rect original_roi;
  int edge_filter;
  bool right_border;
  bool bottom_border;
  double scale1;
  int shiftx, shifty;
  double scale2;
};

// =============================================================================
// Image resizing functions using bilinear interpolation
// =============================================================================

/// Resize an image using bilinear interpolation
///
/// \param src Source image
/// \param dst_width Destination width
/// \param dst_height Destination height
/// \returns Resized image
VIAME_CORE_EXPORT
kv::image
resize_image_bilinear(
  const kv::image& src,
  size_t dst_width,
  size_t dst_height );

/// Scale image maintaining aspect ratio
///
/// \param src Source image
/// \param width Maximum width
/// \param height Maximum height
/// \param pad If true, pad the result to exactly width x height
/// \param scale_out Output parameter for the scale factor applied
/// \returns Scaled image
VIAME_CORE_EXPORT
kv::image
scale_image_maintaining_ar(
  const kv::image& src,
  int width,
  int height,
  bool pad,
  double& scale_out );

/// Format image according to rescale option
///
/// \param src Source image
/// \param option Rescale option
/// \param scale_factor Scale factor for SCALE/CHIP modes
/// \param width Target width
/// \param height Target height
/// \param pad Whether to pad the result
/// \param scale_out Output parameter for the scale factor applied
/// \returns Formatted image
VIAME_CORE_EXPORT
kv::image
format_image(
  const kv::image& src,
  rescale_option option,
  double scale_factor,
  int width,
  int height,
  bool pad,
  double& scale_out );

// =============================================================================
// Detection manipulation functions
// =============================================================================

/// Rescale detections from chip coordinates to original image coordinates
VIAME_CORE_EXPORT
kv::detected_object_set_sptr
rescale_detections(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  double chip_edge_max_prob );

/// Prepare image regions for windowed processing
VIAME_CORE_EXPORT
void
prepare_image_regions(
  const kv::image& image,
  const window_settings& settings,
  std::vector< kv::image >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties );

/// Scale detections by region properties (inverse transform)
VIAME_CORE_EXPORT
void scale_detections(
  kv::detected_object_set_sptr& detections,
  const windowed_region_prop& region_info );

/// Scale detections to fit within a region
VIAME_CORE_EXPORT
kv::detected_object_set_sptr
scale_detections_to_region(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info );

/// Scale detections to region with mapping to original detections
VIAME_CORE_EXPORT
void
scale_detections_to_region_with_mapping(
  const kv::detected_object_set_sptr detections,
  const windowed_region_prop& region_info,
  std::vector< kv::detected_object_sptr >& original_detections,
  std::vector< kv::detected_object_sptr >& scaled_detections );

/// Separate detections that touch image boundaries from interior detections
VIAME_CORE_EXPORT
void
separate_boundary_detections(
  const kv::detected_object_set_sptr detections,
  int region_width,
  int region_height,
  kv::detected_object_set_sptr& boundary_detections,
  kv::detected_object_set_sptr& interior_detections );

// =============================================================================
// Image cropping utility
// =============================================================================

/// Crop a region from an image
///
/// \param src Source image
/// \param roi Region of interest to crop
/// \returns Cropped image (copy of the data)
VIAME_CORE_EXPORT
kv::image
crop_image(
  const kv::image& src,
  const image_rect& roi );

} // end namespace viame

#endif /* VIAME_CORE_WINDOWED_UTILS_H */
