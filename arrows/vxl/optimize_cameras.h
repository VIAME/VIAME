/*ckwg +29
 * Copyright 2014-2016, 2019-2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
