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
 * \brief Implementation of Ceres camera options
 */

#include "camera_options.h"

using namespace kwiver::vital;


namespace kwiver {
namespace arrows {
namespace ceres {

/// Constructor
camera_options
::camera_options()
  : optimize_focal_length(true),
    optimize_aspect_ratio(false),
    optimize_principal_point(false),
    optimize_skew(false),
    lens_distortion_type(NO_DISTORTION),
    optimize_dist_k1(true),
    optimize_dist_k2(false),
    optimize_dist_k3(false),
    optimize_dist_p1_p2(false),
    optimize_dist_k4_k5_k6(false),
    camera_intrinsic_share_type(AUTO_SHARE_INTRINSICS)
{
}

/// Copy Constructor
camera_options
::camera_options(const camera_options& other)
  : optimize_focal_length(other.optimize_focal_length),
    optimize_aspect_ratio(other.optimize_aspect_ratio),
    optimize_principal_point(other.optimize_principal_point),
    optimize_skew(other.optimize_skew),
    lens_distortion_type(other.lens_distortion_type),
    optimize_dist_k1(other.optimize_dist_k1),
    optimize_dist_k2(other.optimize_dist_k2),
    optimize_dist_k3(other.optimize_dist_k3),
    optimize_dist_p1_p2(other.optimize_dist_p1_p2),
    optimize_dist_k4_k5_k6(other.optimize_dist_k4_k5_k6),
    camera_intrinsic_share_type(other.camera_intrinsic_share_type)
{
}


/// populate the config block with options
void
camera_options
::get_configuration(config_block_sptr config) const
{
  config->set_value("optimize_focal_length", this->optimize_focal_length,
                    "Include focal length parameters in bundle adjustment.");
  config->set_value("optimize_aspect_ratio", this->optimize_aspect_ratio,
                    "Include aspect ratio parameters in bundle adjustment.");
  config->set_value("optimize_principal_point", this->optimize_principal_point,
                    "Include principal point parameters in bundle adjustment.");
  config->set_value("optimize_skew", this->optimize_skew,
                    "Include skew parameters in bundle adjustment.");
  config->set_value("lens_distortion_type", this->lens_distortion_type,
                    "Lens distortion model to use."
                    + ceres_options< ceres::LensDistortionType >());
  config->set_value("optimize_dist_k1", this->optimize_dist_k1,
                    "Include radial lens distortion parameter k1 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_k2", this->optimize_dist_k2,
                    "Include radial lens distortion parameter k2 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_k3", this->optimize_dist_k3,
                    "Include radial lens distortion parameter k3 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_p1_p2", this->optimize_dist_p1_p2,
                    "Include tangential lens distortion parameters "
                    "p1 and p2 in bundle adjustment.");
  config->set_value("optimize_dist_k4_k5_k6", this->optimize_dist_k4_k5_k6,
                    "Include radial lens distortion parameters "
                    "k4, k5, and k6 in bundle adjustment.");
  config->set_value("camera_intrinsic_share_type", this->camera_intrinsic_share_type,
                    "Determines how to share intrinsics across cameras.\n"
                    "AUTO shares intrinsics between cameras with a common camera_intrinsic_sptr\n"
                    "COMMON enforces that all cameras share common intrinsics\n"
                    "UNIQUE enforces that each camera has its own intrinsics parameters."
                    + ceres_options< ceres::CameraIntrinsicShareType >());
}


/// set the member variables from the config block
void
camera_options
::set_configuration(config_block_sptr config)
{
#define GET_VALUE(vtype, vname) \
  this->vname = config->get_value< vtype >(#vname, this->vname);

  GET_VALUE(bool, optimize_focal_length);
  GET_VALUE(bool, optimize_aspect_ratio);
  GET_VALUE(bool, optimize_principal_point);
  GET_VALUE(bool, optimize_skew);
  GET_VALUE(bool, optimize_dist_k1);
  GET_VALUE(bool, optimize_dist_k2);
  GET_VALUE(bool, optimize_dist_k3);
  GET_VALUE(bool, optimize_dist_p1_p2);
  GET_VALUE(bool, optimize_dist_k4_k5_k6);
  GET_VALUE(ceres::LensDistortionType, lens_distortion_type);
  GET_VALUE(ceres::CameraIntrinsicShareType, camera_intrinsic_share_type);
#undef GET_VALUE
}


/// enumerate the intrinsics held constant
std::vector<int>
camera_options
::enumerate_constant_intrinsics() const
{
  std::vector<int> constant_intrinsics;

  // number of lens distortion parameters in the selected model
  const unsigned int ndp = num_distortion_params(this->lens_distortion_type);

  if (!this->optimize_focal_length)
  {
    constant_intrinsics.push_back(0);
  }
  if (!this->optimize_principal_point)
  {
    constant_intrinsics.push_back(1);
    constant_intrinsics.push_back(2);
  }
  if (!this->optimize_aspect_ratio)
  {
    constant_intrinsics.push_back(3);
  }
  if (!this->optimize_skew)
  {
    constant_intrinsics.push_back(4);
  }
  if (!this->optimize_dist_k1 && ndp > 0)
  {
    constant_intrinsics.push_back(5);
  }
  if (!this->optimize_dist_k2 && ndp > 1)
  {
    constant_intrinsics.push_back(6);
  }
  if (!this->optimize_dist_p1_p2 && ndp > 3)
  {
    constant_intrinsics.push_back(7);
    constant_intrinsics.push_back(8);
  }
  if (!this->optimize_dist_k3 && ndp > 4)
  {
    constant_intrinsics.push_back(9);
  }
  if (!this->optimize_dist_k4_k5_k6 && ndp > 7)
  {
    constant_intrinsics.push_back(10);
    constant_intrinsics.push_back(11);
    constant_intrinsics.push_back(12);
  }
  return constant_intrinsics;
}


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
