/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Internal header for helper functions for Ceres camera options
 */

#ifndef KWIVER_ARROWS_CERES_CAMERA_OPTIONS_H_
#define KWIVER_ARROWS_CERES_CAMERA_OPTIONS_H_

#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <arrows/ceres/types.h>

namespace kwiver {
namespace arrows {
namespace ceres {

  /// Private implementation class
class camera_options
{
public:
  /// Constructor
  camera_options();

  /// Copy Constructor
  camera_options(const camera_options& other);

  /// populate the config block with options
  void get_configuration(vital::config_block_sptr config) const;

  /// set the member variables from the config block
  void set_configuration(vital::config_block_sptr config);

  /// enumerate the intrinsics held constant
  /**
   * Based on the settings of the boolean optimization switches
   * poplulate a vector of indices marking which intrinsic camera
   * paramaters are held constant.  Indices are:
   *   - \b 0 : focal length
   *   - \b 1 : principal point X
   *   - \b 2 : principal point Y
   *   - \b 3 : aspect ratio
   *   - \b 4 : skew
   *   - \b 5 : radial distortion (k1)
   *   - \b 6 : radial distortion (k2)
   *   - \b 7 : tangential distortion (p1)
   *   - \b 8 : tangential distortion (p2)
   *   - \b 9 : radial distortion (k3)
   *   - \b 10 : radial distortion (k4)
   *   - \b 11 : radial distortion (k5)
   *   - \b 12 : radial distortion (k6)
   */
  std::vector<int> enumerate_constant_intrinsics() const;

  /// option to optimize the focal length
  bool optimize_focal_length;
  /// option to optimize aspect ratio
  bool optimize_aspect_ratio;
  /// option to optimize principal point
  bool optimize_principal_point;
  /// option to optimize skew
  bool optimize_skew;
  /// the lens distortion model to use
  LensDistortionType lens_distortion_type;
  /// option to optimize radial distortion parameter k1
  bool optimize_dist_k1;
  /// option to optimize radial distortion parameter k2
  bool optimize_dist_k2;
  /// option to optimize radial distortion parameter k3
  bool optimize_dist_k3;
  /// option to optimize tangential distortions parameters p1, p2
  bool optimize_dist_p1_p2;
  /// option to optimize radial distortion parameters k4, k5, k6
  bool optimize_dist_k4_k5_k6;
  /// the type of sharing of intrinsics between cameras to use
  CameraIntrinsicShareType camera_intrinsic_share_type;
};


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_CERES_CAMERA_OPTIONS_H_
