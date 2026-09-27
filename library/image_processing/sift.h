/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief SIFT, ported from OpenCV's features module
///
/// `ocv_SIFT` was the last of the feature detectors still reaching for cv2,
/// and it is reached by seven files: the registration utilities, the
/// multimodal registration, the homography IOU tracker and colmap's
/// reconstruction all ask for SIFT keypoints and descriptors. This is the
/// algorithm rather than a call into cv2, so those callers can come off it.
///
/// Like `surf.h` beside it, this is a **port** and not a reimplementation from
/// Lowe's paper: the constants, the thresholds, the interpolation, the
/// histogram layout and the descriptor normalisation are OpenCV's, so the
/// output matches theirs rather than merely being correct SIFT. Its licence and
/// authorship are recorded at the top of sift.cxx.
///
/// Patent: US6711293 covered SIFT and expired in March 2020, which is why
/// OpenCV moved it out of `xfeatures2d` and into the main modules. Unlike SURF
/// there is no non-free build to worry about -- every wheel carries
/// `cv2.SIFT_create` -- so this exists to remove the cv2 dependency, not to
/// restore a missing feature.

#ifndef VIAME_IMAGE_PROCESSING_SIFT_H
#define VIAME_IMAGE_PROCESSING_SIFT_H

#include "viame_image_processing_export.h"

#include <viame/core_types/image.h>

#include <cstdint>
#include <vector>

namespace viame {

namespace sift {

/// What `cv::SIFT::create` takes, under the names the `ocv_SIFT` config keys
/// already use.
struct settings
{
  /// The best this many by response, or all of them at zero.
  int n_features = 0;
  int n_octave_layers = 3;
  double contrast_threshold = 0.04;
  double edge_threshold = 10.0;
  double sigma = 1.6;
};

/// A `cv::KeyPoint`, with the fields SIFT fills.
struct keypoint
{
  float x = 0.0f;
  float y = 0.0f;
  float size = 0.0f;
  float angle = -1.0f;
  float response = 0.0f;
  /// The octave, layer and sub-layer offset packed as `cv::KeyPoint` packs
  /// them: octave in the low byte, layer in the next, and
  /// `round( (offset + 0.5) * 255 )` above that. `detect_and_compute` needs it
  /// back in this form to describe a keypoint it did not detect.
  int octave = 0;
};

/// 128, always: four by four spatial bins of eight orientations.
VIAME_IMAGE_PROCESSING_EXPORT
int descriptor_size();

/// Detect keypoints and, when \p descriptors is not null, describe them.
///
/// With \p use_provided_keypoints the detector is skipped and \p keypoints is
/// described as given, which is what `extract_descriptors` needs; each one's
/// `octave` has to carry the packed octave and layer, since that is what says
/// which level of the pyramid to sample.
///
/// \p descriptors is laid out row-major, 128 floats per keypoint, holding
/// whole numbers from 0 to 255 -- OpenCV's `CV_32F` descriptor type, which is
/// the byte descriptor without the cast.
VIAME_IMAGE_PROCESSING_EXPORT
void detect_and_compute(
  viame::image_of< uint8_t > const& image,
  settings const& config,
  std::vector< keypoint >& keypoints,
  std::vector< float >* descriptors,
  bool use_provided_keypoints = false );

} // namespace sift

} // namespace viame

#endif
