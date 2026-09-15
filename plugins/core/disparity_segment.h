/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_DISPARITY_SEGMENT_H
#define VIAME_CORE_DISPARITY_SEGMENT_H

#include "viame_core_export.h"

#include <utility>
#include <vector>

namespace viame { namespace core {

/// Fit disparity, not depth, along a rectified image segment. For a straight
/// 3D segment, disparity is affine in the image interpolation fraction.
/// Samples are (fraction in [0,1], disparity in pixels). Missing/invalid
/// samples count against max_outliers, as do residuals exceeding max_error.
/// Requires a strict majority, at least three inliers, and support spanning
/// at least half the segment. On failure the output arguments are unchanged.
VIAME_CORE_EXPORT
bool fit_disparity_segment(
  const std::vector< std::pair< double, double > >& samples,
  int requested_samples, int max_outliers, double max_error,
  double& head_disparity, double& tail_disparity );

} } // namespace viame::core
#endif
