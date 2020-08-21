/*ckwg +29
 * Copyright 2017, 2019-2020 by Kitware, Inc.
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
 * \brief OCV estimate_pnp algorithm impl interface
 */

#ifndef KWIVER_ARROWS_OCV_ESTIMATE_PNP_H_
#define KWIVER_ARROWS_OCV_ESTIMATE_PNP_H_


#include <vital/vital_config.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/estimate_pnp.h>


namespace kwiver {
namespace arrows {
namespace ocv {

/// A class that uses OpenCV to estimate a camera's pose from 3D feature
/// and point projection pairs.
class KWIVER_ALGO_OCV_EXPORT estimate_pnp
  : public vital::algo::estimate_pnp
{
public:
  PLUGIN_INFO( "ocv",
               "Estimate camera pose with perspective N point method")

    /// Constructor
  estimate_pnp();

  /// Destructor
  virtual ~estimate_pnp();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

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
  estimate(const std::vector<vital::vector_2d>& pts2d,
           const std::vector<vital::vector_3d>& pts3d,
           const kwiver::vital::camera_intrinsics_sptr cal,
           std::vector<bool>& inliers) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};


} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver


#endif
