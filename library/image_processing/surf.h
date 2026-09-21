/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief SURF, ported from OpenCV's xfeatures2d
///
/// `ocv_SURF` is `cv::xfeatures2d::SURF`, and every opencv-python wheel is
/// built with `OPENCV_ENABLE_NONFREE` off, so cv2 registers the name and then
/// raises when a pipeline runs it. Seven shipped configs select it -- the
/// three `utility_register_frames*`, `common_image_stabilizer` and the three
/// `common_sea_lion_stabilizer_*` -- and those configs are identical to the
/// ones `main` ships, so this branch implements the algorithm rather than
/// editing them.
///
/// This is a port of OpenCV's implementation, not a reimplementation from the
/// paper: the constants, the filter sizes, the sampling steps and the
/// descriptor layout are all theirs, so the output matches. Its licence and
/// authorship are recorded at the top of surf.cxx.
///
/// Patent: SURF is patented, which is why the wheels leave it out. VIAME
/// already ships it on `main` through fletch's non-free OpenCV build; this
/// changes where the code lives, not whether VIAME carries it.

#ifndef VIAME_IMAGE_PROCESSING_SURF_H
#define VIAME_IMAGE_PROCESSING_SURF_H

#include "viame_image_processing_export.h"

#include <viame/core_types/image.h>

#include <cstdint>
#include <vector>

namespace viame {

namespace surf {

/// What `cv::xfeatures2d::SURF_create` takes, under the names the
/// `ocv_SURF` config keys already use.
struct settings
{
  double hessian_threshold = 100.0;
  int n_octaves = 4;
  int n_octaves_layers = 3;
  bool extended = false;
  bool upright = false;
};

/// A `cv::KeyPoint`, with only the fields SURF fills.
struct keypoint
{
  float x = 0.0f;
  float y = 0.0f;
  float size = 0.0f;
  float angle = -1.0f;
  float response = 0.0f;
  int octave = 0;
  /// The sign of the Hessian trace, which `cv::KeyPoint` carries in class_id.
  int laplacian = 0;
};

/// 64, or 128 when `extended`.
VIAME_IMAGE_PROCESSING_EXPORT
int descriptor_size( settings const& config );

/// Detect keypoints and, when \p descriptors is not null, describe them.
///
/// With \p use_provided_keypoints the detector is skipped and \p keypoints is
/// described as given -- which is what `extract_descriptors` needs. Either way
/// the orientation pass runs and rewrites each keypoint's angle, and keypoints
/// too close to a border to describe are dropped from both outputs together.
///
/// \p descriptors is laid out row-major, `descriptor_size()` floats per
/// keypoint, and is sized by this call.
VIAME_IMAGE_PROCESSING_EXPORT
void detect_and_compute(
  viame::image_of< uint8_t > const& image,
  settings const& config,
  std::vector< keypoint >& keypoints,
  std::vector< float >* descriptors,
  bool use_provided_keypoints = false );

} // namespace surf

} // namespace viame

#endif
