// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief estimate_fundamental_matrix algorithm definition
 */

#ifndef VITAL_ALGO_ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define VITAL_ALGO_ESTIMATE_FUNDAMENTAL_MATRIX_H_

#include <vital/vital_config.h>

#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_set.h>
#include <vital/types/match_set.h>
#include <vital/types/fundamental_matrix.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for estimating a fundamental matrix from matching 2D points
class VITAL_ALGO_EXPORT estimate_fundamental_matrix
  : public kwiver::vital::algorithm_def<estimate_fundamental_matrix>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "estimate_fundamental_matrix"; }

  /// Estimate an fundamental matrix from corresponding features
  /**
   * \param [in]  feat1 the set of all features from the first image
   * \param [in]  feat2 the set of all features from the second image
   * \param [in]  matches the set of correspondences between \a feat1 and \a feat2
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  virtual
  kwiver::vital::fundamental_matrix_sptr
  estimate(const kwiver::vital::feature_set_sptr feat1,
           const kwiver::vital::feature_set_sptr feat2,
           const kwiver::vital::match_set_sptr matches,
           std::vector<bool>& inliers,
           double inlier_scale = 1.0) const;

  /// Estimate an fundamental matrix from corresponding points
  /**
   * \param [in]  pts1 the vector or corresponding points from the first image
   * \param [in]  pts2 the vector of corresponding points from the second image
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  virtual
  kwiver::vital::fundamental_matrix_sptr
  estimate(const std::vector<kwiver::vital::vector_2d>& pts1,
           const std::vector<kwiver::vital::vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale = 1.0) const = 0;

protected:
  estimate_fundamental_matrix();

};

/// Shared pointer type of base estimate_fundamental_matrix algorithm definition class
typedef std::shared_ptr<estimate_fundamental_matrix> estimate_fundamental_matrix_sptr;

} } } // end namespace

#endif // VITAL_ALGO_ESTIMATE_FUNDAMENTAL_MATRIX_H_
