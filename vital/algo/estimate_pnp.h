// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief estimate_pnp algorithm definition
 */

#ifndef VITAL_ALGO_ESTIMATE_PNP_H_
#define VITAL_ALGO_ESTIMATE_PNP_H_

#include <vital/vital_config.h>

#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_set.h>
#include <vital/types/match_set.h>
#include <vital/types/camera_perspective.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class to estimate a camera's pose from 3D feature
/// and point projection pairs.

class VITAL_ALGO_EXPORT estimate_pnp
  : public kwiver::vital::algorithm_def<estimate_pnp>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "estimate_pnp"; }

  /// Estimate the camera's pose from the 3D points and their corresponding projections
  /**
   * \param [in]  pts2d 2d projections of pts3d in the same order as pts3d
   * \param [in]  pts3d 3d landmarks in the same order as pts2d.  Both must be same size.
   * \param [in]  cal the intrinsic parameters of the camera
   * \param [out] inliers for each point, the value is true if
   *                      this pair is an inlier to the estimate
   */
  virtual
  kwiver::vital::camera_perspective_sptr
  estimate(const std::vector<vector_2d>& pts2d,
           const std::vector<vector_3d>& pts3d,
           const kwiver::vital::camera_intrinsics_sptr cal,
           std::vector<bool>& inliers) const = 0;

protected:
  estimate_pnp();

};

/// Shared pointer type of base estimate_homography algorithm definition class
typedef std::shared_ptr<estimate_pnp> estimate_pnp_sptr;

} } } // end namespace

#endif // VITAL_ALGO_ESTIMATE_PNP_H_
