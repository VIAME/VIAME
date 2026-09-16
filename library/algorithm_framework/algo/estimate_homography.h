// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief estimate_homography algorithm definition

#ifndef VITAL_ALGO_ESTIMATE_HOMOGRAPHY_H_
#define VITAL_ALGO_ESTIMATE_HOMOGRAPHY_H_

#include <viame/algorithm_framework/viame_compiler_config.h>

#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/homography.h>
#include <viame/core_types/match_set.h>
#include <viame/core_types/matrix.h>

namespace viame {

namespace algo {

/// An abstract base class for estimating a homography from matching 2D points
class VIAME_ALGO_EXPORT estimate_homography
  : public viame::algorithm
{
public:
  estimate_homography();
  PLUGGABLE_INTERFACE( estimate_homography );
  /// Estimate a homography matrix from corresponding features
  ///
  /// If estimation fails, a NULL-containing sptr is returned
  ///
  /// \param [in]  feat1 the set of all features from the source image
  /// \param [in]  feat2 the set of all features from the destination image
  /// \param [in]  matches the set of correspondences between \a feat1 and \a
  /// feat2
  /// \param [out] inliers for each match in \a matcher, the value is true if
  ///                      this pair is an inlier to the homography estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual viame::homography_sptr
  estimate(
    viame::feature_set_sptr feat1,
    viame::feature_set_sptr feat2,
    viame::match_set_sptr matches,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const;

  /// Estimate a homography matrix from corresponding points
  ///
  /// If estimation fails, a NULL-containing sptr is returned
  ///
  /// \param [in]  pts1 the vector or corresponding points from the source image
  /// \param [in]  pts2 the vector of corresponding points from the destination
  /// image
  /// \param [out] inliers for each point pair, the value is true if
  ///                      this pair is an inlier to the homography estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual viame::homography_sptr
  estimate(
    const std::vector< viame::vector_2d >& pts1,
    const std::vector< viame::vector_2d >& pts2,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const = 0;
};

/// Shared pointer type of base estimate_homography algorithm definition class
typedef std::shared_ptr< estimate_homography > estimate_homography_sptr;

} // namespace algo

} // namespace viame

#endif // VITAL_ALGO_ESTIMATE_HOMOGRAPHY_H_
