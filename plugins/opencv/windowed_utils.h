/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_WINDOWED_UTILS_H
#define VIAME_OPENCV_WINDOWED_UTILS_H

// Include core windowed utilities for shared types and functions
// This provides: rescale_option, window_settings, image_rect, windowed_region_prop,
// and all detection manipulation functions (rescale_detections, scale_detections, etc.)
#include "../core/windowed_utils.h"

#include "viame_opencv_export.h"

#include <viame/core_types/image.h>

#include <vector>

namespace viame {

// =============================================================================
// Chipping an image up for a windowed detector
//
// `cv::Mat` and `cv::resize` until P7-T04b; `vital::image` and
// `image_ops::resize` since. The rectangles were always `image_rect`, which
// is why there is no conversion helper here any more.
// =============================================================================

/// Scale an image to fit, keeping its aspect ratio
///
/// \param src Source image
/// \param dst Destination image (output)
/// \param width Maximum width
/// \param height Maximum height
/// \param pad If true, pad the result to exactly width x height
/// \returns Scale factor applied
VIAME_OPENCV_EXPORT
double
scale_image_maintaining_ar(
  const kwiver::vital::image& src,
  kwiver::vital::image& dst,
  int width,
  int height,
  bool pad = false );

/// Format an image according to a rescale option
///
/// \param src Source image
/// \param dst Destination image (output)
/// \param option Rescale option
/// \param scale_factor Scale factor for SCALE/CHIP modes
/// \param width Target width
/// \param height Target height
/// \param pad Whether to pad the result
/// \returns Scale factor applied
VIAME_OPENCV_EXPORT
double
format_image(
  const kwiver::vital::image& src,
  kwiver::vital::image& dst,
  rescale_option option,
  double scale_factor,
  int width,
  int height,
  bool pad = false );

/// The \p rect region of \p image, as a new image
///
/// `cv::Mat`'s region-of-interest operator, which the trainer used to chip
/// with. Unlike OpenCV's this copies rather than aliasing, which is what the
/// chip writer wants anyway.
VIAME_OPENCV_EXPORT
kwiver::vital::image
crop_region(
  const kwiver::vital::image& image,
  const image_rect& rect );

// `prepare_image_regions` is declared by `../core/windowed_utils.h`, above,
// and defined once -- there. Declaring it here as well gave one symbol two
// definitions in two libraries; see the note where the definition used to be.

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_UTILS_H */
