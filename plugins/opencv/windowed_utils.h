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

// Include core windowed utilities for shared types and functions
// This provides: rescale_option, window_settings, image_rect, windowed_region_prop,
// and all detection manipulation functions (rescale_detections, scale_detections, etc.)
#include "../core/windowed_utils.h"

#include "viame_opencv_export.h"

#include <opencv2/core/core.hpp>

namespace viame {

// =============================================================================
// Conversion utilities between image_rect and cv::Rect
// =============================================================================

/// Convert image_rect to cv::Rect
inline cv::Rect to_cv_rect( const image_rect& r )
{
  return cv::Rect( r.x, r.y, r.width, r.height );
}

/// Convert cv::Rect to image_rect
inline image_rect from_cv_rect( const cv::Rect& r )
{
  return image_rect( r.x, r.y, r.width, r.height );
}

// =============================================================================
// OpenCV-specific image processing functions
// These use cv::resize which is faster than the core bilinear implementation
// =============================================================================

/// Scale image maintaining aspect ratio using OpenCV
///
/// \param src Source cv::Mat image
/// \param dst Destination cv::Mat image (output)
/// \param width Maximum width
/// \param height Maximum height
/// \param pad If true, pad the result to exactly width x height
/// \returns Scale factor applied
VIAME_OPENCV_EXPORT
double
scale_image_maintaining_ar(
  const cv::Mat& src,
  cv::Mat& dst,
  int width,
  int height,
  bool pad = false );

/// Format image according to rescale option using OpenCV
///
/// \param src Source cv::Mat image
/// \param dst Destination cv::Mat image (output)
/// \param option Rescale option
/// \param scale_factor Scale factor for SCALE/CHIP modes
/// \param width Target width
/// \param height Target height
/// \param pad Whether to pad the result
/// \returns Scale factor applied
VIAME_OPENCV_EXPORT
double
format_image(
  const cv::Mat& src,
  cv::Mat& dst,
  rescale_option option,
  double scale_factor,
  int width,
  int height,
  bool pad = false );

/// Prepare image regions for windowed processing using OpenCV
///
/// This function breaks up an image into regions based on window settings
/// and returns both the image regions and their properties for detection
/// coordinate transformation.
///
/// \param image Input cv::Mat image
/// \param settings Window settings configuration
/// \param regions_to_process Output vector of cv::Mat image regions
/// \param region_properties Output vector of region properties for coordinate transforms
VIAME_OPENCV_EXPORT
void
prepare_image_regions(
  const cv::Mat& image,
  const window_settings& settings,
  std::vector< cv::Mat >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties );

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_UTILS_H */
