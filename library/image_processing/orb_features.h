/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `ocv_ORB`, over the ported algorithm in orb.h
///
/// The name came out in P5-T04, when `arrows/ocv` was split across the
/// functional libraries and only what a shipped config selected came across --
/// `tests/baseline/removed.json` records that. Nothing selected it, but five
/// pipeline configs *name* it in the comment above `feature_detector:type` and
/// carry an inert `block feature_detector:ocv_ORB`, so a user who followed the
/// comment got an implementation that would not resolve. It is back over
/// `orb.h`, which is VIAME's own ORB rather than a call into cv2.
///
/// The config keys, their defaults and their descriptions are the ones the
/// OpenCV arrow registered and `registry.json` records, including the two
/// places they differ from `cv::ORB::create`'s own defaults: `n_levels` is
/// **9** rather than 8, and `score_type` is the integer OpenCV's enumeration
/// uses rather than a name.

#ifndef VIAME_IMAGE_PROCESSING_ORB_FEATURES_H
#define VIAME_IMAGE_PROCESSING_ORB_FEATURES_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>
#include <viame/core_types/feature_set.h>

#include <vector>

namespace viame {

namespace kv = viame;

/// A feature set that also remembers which pyramid level each keypoint came
/// from.
///
/// ORB needs it and `vital::feature` has nowhere to put it. The level decides
/// which image the descriptor is sampled from and by how much the keypoint's
/// position is scaled back down, so without it every keypoint is described at
/// the base of the pyramid -- scale invariant in name only. The same problem
/// `sift_feature_set` solves, and solved the same way: `extract` recovers the
/// levels when the set it is handed is one of these and uses zero when it is
/// not, which is what cv2 does with the same input.
///
/// This is deliberately not shared with `sift_feature_set`. SIFT's field is a
/// **packed** octave -- octave, layer and sub-layer offset in one int -- and
/// ORB's is the level itself; a common class would have to say which, and the
/// two algorithms would then be free to disagree about it.
class VIAME_IMAGE_PROCESSING_EXPORT orb_feature_set
  : public kv::simple_feature_set
{
public:
  orb_feature_set( std::vector< kv::feature_sptr > const& features,
                   std::vector< int > const& levels )
    : kv::simple_feature_set( features ), levels_( levels ) {}

  std::vector< int > const& levels() const { return levels_; }

private:
  std::vector< int > levels_;
};

#define VIAME_ORB_PARAMS \
    PARAM_DEFAULT( \
      n_features, int, \
      "The maximum number of features to retain", \
      500 ), \
    PARAM_DEFAULT( \
      scale_factor, double, \
      "Pyramid decimation ratio, greater than 1. scaleFactor==2 means the " \
      "classical pyramid, where each next level has 4x less pixels than the " \
      "previous, but such a big scale factor will degrade feature matching " \
      "scores dramatically. On the other hand, too close to 1 scale factor " \
      "will mean that to cover certain scale range you will need more " \
      "pyramid levels and so the speed will suffer. Narrowed to float " \
      "before use, as cv::ORB::create narrows it: asking for 1.2 is asking " \
      "for 1.2000000476837158, and every level's scale follows from it.", \
      1.2 ), \
    PARAM_DEFAULT( \
      n_levels, int, \
      "The number of pyramid levels. The smallest level will have linear " \
      "size equal to input_image_linear_size/pow(scale_factor, n_levels).", \
      9 ), \
    PARAM_DEFAULT( \
      edge_threshold, int, \
      "This is size of the border where the features are not detected. It " \
      "should roughly match the patch_size parameter.", \
      31 ), \
    PARAM_DEFAULT( \
      first_level, int, \
      "It should be 0 in the current implementation.", \
      0 ), \
    PARAM_DEFAULT( \
      wta_k, int, \
      "The number of points that produce each element of the oriented BRIEF " \
      "descriptor. The default value 2 means the BRIEF where we take a " \
      "random point pair and compare their brightnesses, so we get 0/1 " \
      "response. Other possible values are 3 and 4, which VIAME's own ORB " \
      "does not implement: both draw their sampling pattern from cv::RNG, so " \
      "a descriptor built from a different pattern is not comparable, and " \
      "the two-bit output would need a Hamming-2 matcher that nothing here " \
      "offers.", \
      2 ), \
    PARAM_DEFAULT( \
      score_type, int, \
      "The default HARRIS_SCORE (value=cv::ORB::HARRIS_SCORE) means that " \
      "Harris algorithm is used to rank features (the score is written to " \
      "KeyPoint::score and is used to retain best n_features features); " \
      "FAST_SCORE (value=cv::ORB::FAST_SCORE) is alternative value of the " \
      "parameter that produces slightly less stable key-points, but it is a " \
      "little faster to compute.", \
      0 ), \
    PARAM_DEFAULT( \
      patch_size, int, \
      "Size of the patch used by the oriented BRIEF descriptor. Of course, " \
      "on smaller pyramid layers the perceived image area covered by a " \
      "feature will be larger. Only 31 is implemented, the size the learned " \
      "sampling pattern was built for.", \
      31 ), \
    PARAM_DEFAULT( \
      fast_threshold, int, \
      "The contrast a ring pixel needs before it counts towards a FAST " \
      "corner, from 0 to 255.", \
      20 )

/// @brief Detect ORB keypoints.
class VIAME_IMAGE_PROCESSING_EXPORT detect_features_ORB
  : public kv::algo::detect_features
{
public:
  PLUGGABLE_IMPL_NAMED(
    detect_features_ORB,
    "ocv_ORB",
    "OpenCV feature detection via the ORB algorithm",
    VIAME_ORB_PARAMS )

  virtual ~detect_features_ORB();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::feature_set_sptr detect(
    kv::image_container_sptr image_data,
    kv::image_container_sptr mask ) const override;

private:
  void initialize() override;
};

/// @brief Describe features with ORB.
class VIAME_IMAGE_PROCESSING_EXPORT extract_descriptors_ORB
  : public kv::algo::extract_descriptors
{
public:
  PLUGGABLE_IMPL_NAMED(
    extract_descriptors_ORB,
    "ocv_ORB",
    "OpenCV feature description via the ORB algorithm",
    VIAME_ORB_PARAMS )

  virtual ~extract_descriptors_ORB();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::descriptor_set_sptr extract(
    kv::image_container_sptr image_data,
    kv::feature_set_sptr& features,
    kv::image_container_sptr image_mask ) const override;

private:
  void initialize() override;
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_ORB_FEATURES_H
