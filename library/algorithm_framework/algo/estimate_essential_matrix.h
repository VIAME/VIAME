// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief estimate_essential_matrix algorithm definition

#ifndef VITAL_ALGO_ESTIMATE_ESSENTIAL_MATRIX_H_
#define VITAL_ALGO_ESTIMATE_ESSENTIAL_MATRIX_H_

#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/camera_intrinsics.h>
#include <viame/core_types/essential_matrix.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/match_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for estimating an essential matrix from matching 2D
/// points
class VITAL_ALGO_EXPORT estimate_essential_matrix
  : public kwiver::vital::algorithm
{
public:
  estimate_essential_matrix();
  PLUGGABLE_INTERFACE( estimate_essential_matrix );

  /// Estimate an essential matrix from corresponding features
  ///
  /// \param [in]  feat1 the set of all features from the first image
  /// \param [in]  feat2 the set of all features from the second image
  /// \param [in]  matches the set of correspondences between \a feat1 and \a
  /// feat2
  /// \param [in]  cal1 the intrinsic parameters of the first camera
  /// \param [in]  cal2 the intrinsic parameters of the second camera
  /// \param [out] inliers for each point pair, the value is true if
  ///                      this pair is an inlier to the estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual
  kwiver::vital::essential_matrix_sptr
  estimate(
    const kwiver::vital::feature_set_sptr feat1,
    const kwiver::vital::feature_set_sptr feat2,
    const kwiver::vital::match_set_sptr matches,
    const kwiver::vital::camera_intrinsics_sptr cal1,
    const kwiver::vital::camera_intrinsics_sptr cal2,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const;

  /// Estimate an essential matrix from corresponding features
  ///
  /// \param [in]  feat1 the set of all features from the first image
  /// \param [in]  feat2 the set of all features from the second image
  /// \param [in]  matches the set of correspondences between \a feat1 and \a
  /// feat2
  /// \param [in]  cal the intrinsic parameters, same for both cameras
  /// \param [out] inliers for each point pair, the value is true if
  ///                      this pair is an inlier to the estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual
  kwiver::vital::essential_matrix_sptr
  estimate(
    const kwiver::vital::feature_set_sptr feat1,
    const kwiver::vital::feature_set_sptr feat2,
    const kwiver::vital::match_set_sptr matches,
    const kwiver::vital::camera_intrinsics_sptr cal,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const;

  /// Estimate an essential matrix from corresponding points
  ///
  /// \param [in]  pts1 the vector or corresponding points from the first image
  /// \param [in]  pts2 the vector of corresponding points from the second image
  /// \param [in]  cal the intrinsic parameters, same for both cameras
  /// \param [out] inliers for each point pair, the value is true if
  ///                      this pair is an inlier to the estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual
  kwiver::vital::essential_matrix_sptr
  estimate(
    const std::vector< kwiver::vital::vector_2d >& pts1,
    const std::vector< kwiver::vital::vector_2d >& pts2,
    const kwiver::vital::camera_intrinsics_sptr cal,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const;

  /// Estimate an essential matrix from corresponding points
  ///
  /// \param [in]  pts1 the vector or corresponding points from the first image
  /// \param [in]  pts2 the vector of corresponding points from the second image
  /// \param [in]  cal1 the intrinsic parameters of the first camera
  /// \param [in]  cal2 the intrinsic parameters of the second camera
  /// \param [out] inliers for each point pa:wir, the value is true if
  ///                      this pair is an inlier to the estimate
  /// \param [in]  inlier_scale error distance tolerated for matches to be
  /// inliers
  virtual
  kwiver::vital::essential_matrix_sptr
  estimate(
    const std::vector< kwiver::vital::vector_2d >& pts1,
    const std::vector< kwiver::vital::vector_2d >& pts2,
    const kwiver::vital::camera_intrinsics_sptr cal1,
    const kwiver::vital::camera_intrinsics_sptr cal2,
    std::vector< bool >& inliers,
    double inlier_scale = 1.0 ) const = 0;
};

/// Shared pointer type of base estimate_homography algorithm definition class
typedef std::shared_ptr< estimate_essential_matrix >
  estimate_essential_matrix_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_ESTIMATE_ESSENTIAL_MATRIX_H_
