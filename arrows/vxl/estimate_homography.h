// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL homography estimation algorithm
 */

#ifndef KWIVER_ARROWS_VXL_ESTIMATE_HOMOGRAPHY_H_
#define KWIVER_ARROWS_VXL_ESTIMATE_HOMOGRAPHY_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/estimate_homography.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A class that uses RREL in VXL to estimate a homography from matching 2D points
class KWIVER_ALGO_VXL_EXPORT estimate_homography
  : public vital::algo::estimate_homography
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (rrel) to robustly estimate a homography from matched features." )

  // No configuration yet for this class.
  /// \cond DoxygenSuppress
  virtual void set_configuration(vital::config_block_sptr /*config*/) {}
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }
  /// \endcond

  /// Estimate a homography matrix from corresponding points
  /**
   * If estimation fails, a NULL-containing sptr is returned
   *
   * \param [in]  pts1 the vector or corresponding points from the source image
   * \param [in]  pts2 the vector of corresponding points from the destination image
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the homography estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  virtual vital::homography_sptr
  estimate(const std::vector<vital::vector_2d>& pts1,
           const std::vector<vital::vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale = 1.0) const;
  using vital::algo::estimate_homography::estimate;

};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
