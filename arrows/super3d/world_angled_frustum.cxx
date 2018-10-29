/*ckwg +29
 * Copyright 2012-2018 by Kitware, Inc.
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
 * \brief Source file for world_angled_frustum
 */


#include "world_angled_frustum.h"
#include <vnl/vnl_inverse.h>

namespace kwiver {
namespace arrows {
namespace super3d {

/// Constructor
world_angled_frustum
::world_angled_frustum(const vpgl_perspective_camera<double> &cam,
                       const vnl_vector_fixed<double,3>& plane_normal,
                       double min_offset,
                       double max_offset,
                       unsigned int image_width,
                       unsigned int image_height)
: world_space(image_width, image_height),
  min_offset_(min_offset),
  offset_range_(max_offset - min_offset),
  normal_(plane_normal),
  ref_cam_(cam)
{
  const vpgl_calibration_matrix<double> &K = cam.get_calibration();
  KR_inv_ = vnl_inverse(K.get_matrix() * cam.get_rotation().as_matrix());

  vgl_point_3d<double> c = cam.get_camera_center();
  cam_center_ = vnl_double_3(c.x(), c.y(), c.z());

  height_offset_ = dot_product(cam_center_, normal_) - min_offset_;
  denom_vec_ = normal_ * KR_inv_;
}

//*****************************************************************************

/// returns the corner points of an image slice at depth slice.
/// depth slice is a value between 0 and 1 over the depth range
std::vector<vnl_double_3> world_angled_frustum::get_slice(double depth_slice) const
{
  std::vector<vnl_double_3> slice;
  slice.push_back(point_at_depth_on_axis(0.0, 0.0, depth_slice));
  slice.push_back(point_at_depth_on_axis((double)ni_, 0.0, depth_slice));
  slice.push_back(point_at_depth_on_axis((double)ni_, (double)nj_, depth_slice));
  slice.push_back(point_at_depth_on_axis(0.0, (double)nj_, depth_slice));
  return slice;
}

//*****************************************************************************

/// Maps a slice value at image location (i,j) to a depth value
double
world_angled_frustum
::slice_to_depth(double i, double j, double slice) const
{
  return -(height_offset_ - (offset_range_ * slice))
         / dot_product(denom_vec_, vnl_double_3(i, j, 1));
}

//*****************************************************************************

/// Maps a slice value at image location (i,j) to a 3D world point
vnl_double_3
world_angled_frustum
::point_at_depth_on_axis(double i, double j, double slice) const
{
  const double d = slice_to_depth(i, j, slice);
  return KR_inv_ * vnl_double_3(d*i, d*j, d) + cam_center_;
}

//*****************************************************************************

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
