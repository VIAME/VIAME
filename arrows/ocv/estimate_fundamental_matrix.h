// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV estimate_fundamental_matrix algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_ESTIMATE_FUNDAMENTAL_MATRIX_H_
#define KWIVER_ARROWS_OCV_ESTIMATE_FUNDAMENTAL_MATRIX_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/estimate_fundamental_matrix.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class that using OpenCV to estimate a fundamental matrix from matching 2D points
class KWIVER_ALGO_OCV_EXPORT estimate_fundamental_matrix
  : public vital::algo::estimate_fundamental_matrix
{
public:
  PLUGIN_INFO( "ocv",
               "Use OpenCV to estimate a fundimental matrix from feature matches." )

   /// Constructor
  estimate_fundamental_matrix();

  /// Destructor
  virtual ~estimate_fundamental_matrix();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Estimate a fundamental matrix from corresponding points
  /**
   * If estimation fails, a NULL-containing sptr is returned
   *
   * \param [in]  pts1 the vector or corresponding points from the source image
   * \param [in]  pts2 the vector of corresponding points from the destination image
   * \param [out] inliers for each point pair, the value is true if
   *                      this pair is an inlier to the fundamental matrix estimate
   * \param [in]  inlier_scale error distance tolerated for matches to be inliers
   */
  virtual vital::fundamental_matrix_sptr
  estimate(const std::vector<vital::vector_2d>& pts1,
           const std::vector<vital::vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale = 3.0) const;
  using vital::algo::estimate_fundamental_matrix::estimate;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
