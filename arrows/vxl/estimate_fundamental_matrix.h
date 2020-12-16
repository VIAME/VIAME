// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL fundamental matrix estimation algorithm (5 point alg)
 */

#ifndef KWIVER_ARROWS_VXL_ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define KWIVER_ARROWS_VXL_ESTIMATE_FUNDAMENTAL_MATRIX_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/camera_intrinsics.h>

#include <vital/algo/estimate_fundamental_matrix.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A class that uses 5 pt algorithm to estimate an initial xform between 2 pt sets
class KWIVER_ALGO_VXL_EXPORT estimate_fundamental_matrix
  : public vital::algo::estimate_fundamental_matrix
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (vpgl) to estimate a fundamental matrix." )

  /// Constructor
  estimate_fundamental_matrix();

  /// Destructor
  virtual ~estimate_fundamental_matrix();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Estimate an fundamental matrix from corresponding points
  /**
   * \param [in]  pts1 the vector or corresponding points from the first image
   * \param [in]  pts2 the vector of corresponding points from the second image
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  virtual
  vital::fundamental_matrix_sptr
  estimate(const std::vector<vital::vector_2d>& pts1,
           const std::vector<vital::vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale = 1.0) const;
  using vital::algo::estimate_fundamental_matrix::estimate;

  /// Test corresponding points against a fundamental matrix and mark inliers
  /**
   * \param [in]  fm   the fundamental matrix
   * \param [in]  pts1 the vector or corresponding points from the first image
   * \param [in]  pts2 the vector of corresponding points from the second image
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  static void
  mark_inliers(vital::fundamental_matrix_sptr const& fm,
               std::vector<vital::vector_2d> const& pts1,
               std::vector<vital::vector_2d> const& pts2,
               std::vector<bool>& inliers,
               double inlier_scale = 1.0);
private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
