/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief ORB, ported from OpenCV's features module
///
/// The last of the three detectors this branch reaches for. `ocv_ORB` is
/// named as an option by five pipeline configs and was not registered at
/// all, because the arrow that used to provide it came out with OpenCV;
/// `homog_iou_tracker` keeps one branch on cv2 for it, and
/// `tools/calibrate.py` falls back to it when SIFT is unavailable.
///
/// Like `sift.h` and `surf.h` beside it this is a **port**: the pyramid,
/// the thresholds, the sampling pattern and the rounding are OpenCV's, so
/// the output matches theirs rather than merely being correct ORB. There
/// is no patent here -- ORB was written to be the unencumbered answer to
/// SIFT and SURF -- so this exists to remove the dependency.
///
/// **The keypoint order is ours, not cv2's, and it has to be.** ORB culls
/// each level to a feature budget with `std::nth_element` followed by
/// `std::partition`, neither of which specifies the order it leaves behind.
/// The *set* that survives is well defined -- every keypoint whose response
/// reaches the n-th largest, ties included -- and that is reproduced
/// exactly. What cv2 then hands back is whatever its standard library's
/// introselect happened to produce. This returns them in detection order
/// instead: by level, and within a level by row and then column. Nothing
/// downstream can tell the difference, since a matcher pairs row i of the
/// descriptors with keypoint i and both are written in the same order.

#ifndef VIAME_IMAGE_PROCESSING_ORB_H
#define VIAME_IMAGE_PROCESSING_ORB_H

#include "viame_image_processing_export.h"

#include <viame/core_types/image.h>

#include <cstdint>
#include <vector>

namespace viame {

namespace orb {

/// What `cv::ORB::create` takes, under the names the `ocv_ORB` config keys
/// already use.
struct settings
{
  /// The budget, shared out over the levels in a geometric series.
  int n_features = 500;
  /// Ratio between successive pyramid levels.
  ///
  /// A **float**, which is not a detail. `cv::ORB_Impl` stores a double,
  /// but `cv::ORB::create` -- the only way anything constructs one --
  /// takes a float, so a caller asking for 1.2 is really asking for
  /// 1.2000000476837158, and every level's scale and every keypoint's
  /// position follows from it. Storing a true double here would put the
  /// pyramid a tenth of a thousandth of a pixel away from cv2's and make
  /// the descriptors near the edge of a patch disagree.
  float scale_factor = 1.2f;
  int n_levels = 8;
  /// How far from the edge a keypoint has to be. Also the size of the
  /// border the pyramid carries, near enough.
  int edge_threshold = 31;
  /// The level whose scale is one. Levels below it are enlargements of the
  /// source rather than reductions.
  int first_level = 0;
  /// Comparisons per descriptor bit. Only 2 is implemented; see the note in
  /// orb.cxx on what 3 and 4 would need.
  int wta_k = 2;
  /// Harris cornerness re-ranks the FAST corners when true, which is
  /// `ORB::HARRIS_SCORE`; false keeps the FAST suppression score, which is
  /// `ORB::FAST_SCORE`.
  bool harris_score = true;
  /// The descriptor patch. Only 31 is implemented, since any other size
  /// draws its sampling pattern from `cv::RNG`.
  int patch_size = 31;
  int fast_threshold = 20;
};

/// A `cv::KeyPoint`, with the fields ORB fills.
struct keypoint
{
  float x = 0.0f;
  float y = 0.0f;
  float size = 0.0f;
  float angle = -1.0f;
  float response = 0.0f;
  /// The pyramid level, which is what `cv::ORB` puts here -- not SIFT's
  /// packed octave. `detect_and_compute` needs it back to know which level
  /// to sample a keypoint it did not detect.
  int octave = 0;
};

/// 32 for `wta_k == 2`: 256 comparisons, one bit each.
VIAME_IMAGE_PROCESSING_EXPORT
int descriptor_size( settings const& config );

/// Detect keypoints and, when \p descriptors is not null, describe them.
///
/// With \p use_provided_keypoints the detector is skipped and \p keypoints
/// is described as given. Each one's `octave` says which level to sample and
/// its `x` and `y` are in the coordinates of the source image, as they would
/// be coming back from a detection.
///
/// \p descriptors is laid out row-major, `descriptor_size` bytes per
/// keypoint.
VIAME_IMAGE_PROCESSING_EXPORT
void detect_and_compute(
  viame::image_of< uint8_t > const& image,
  settings const& config,
  std::vector< keypoint >& keypoints,
  std::vector< uint8_t >* descriptors,
  bool use_provided_keypoints = false );

} // namespace orb

} // namespace viame

#endif // VIAME_IMAGE_PROCESSING_ORB_H
