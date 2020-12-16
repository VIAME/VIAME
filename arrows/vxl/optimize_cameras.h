// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header defining VXL algorithm implementation of camera optimization.
*/

#ifndef KWIVER_ARROWS_VXL_OPTIMIZE_CAMERAS_H_
#define KWIVER_ARROWS_VXL_OPTIMIZE_CAMERAS_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/optimize_cameras.h>
#include <vital/types/camera_perspective.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace vxl {

class KWIVER_ALGO_VXL_EXPORT optimize_cameras
  : public vital::algo::optimize_cameras
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (vpgl) to optimize camera parameters for fixed "
               "landmarks and tracks." )

  /// \cond DoxygenSuppress
  virtual void set_configuration(vital::config_block_sptr /*config*/) { }
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }
  /// \endcond

  using vital::algo::optimize_cameras::optimize;

  /// Optimize a single camera given corresponding features and landmarks
  /**
   * This function assumes that 2D features viewed by this camera have
   * already been put into correspondence with 3D landmarks by aligning
   * them into two parallel vectors
   *
   * \param[in,out] camera    The camera to optimize.
   * \param[in]     features  The vector of features observed by \p camera
   *                          to use as constraints.
   * \param[in]     landmarks The vector of landmarks corresponding to
   *                          \p features.
   * \param[in]     metadata  The optional metadata to constrain the
   *                          optimization.
   */
  virtual void
  optimize(kwiver::vital::camera_perspective_sptr & camera,
           const std::vector<vital::feature_sptr>& features,
           const std::vector<vital::landmark_sptr>& landmarks,
           kwiver::vital::sfm_constraints_sptr constraints = nullptr) const;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
