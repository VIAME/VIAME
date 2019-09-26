/*ckwg +29
* Copyright 2012-2019 by Kitware, Inc.
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
* \brief Header file for world space defines how the volumes slices the world
*/

#ifndef KWIVER_ARROWS_SUPER3D_WORLD_SPACE_H_
#define KWIVER_ARROWS_SUPER3D_WORLD_SPACE_H_

#include <vital/vital_config.h>

#include <vector>
#include <vil/vil_image_view.h>
#include <vpgl/vpgl_perspective_camera.h>
#include <vnl/vnl_double_3.h>

#include "warp_image.h"


namespace kwiver {
namespace arrows {
namespace super3d {

class world_space
{
public:
  world_space(unsigned int pixel_width, unsigned int pixel_height);
  virtual ~world_space() {}

  /// returns the corner points of an image slice at depth slice.
  /// depth slice is a value between 0 and 1 over the depth range
  virtual std::vector<vnl_double_3> get_slice(double depth_slice) const = 0;

  virtual std::vector<vpgl_perspective_camera<double> >
          warp_cams(const std::vector<vpgl_perspective_camera<double> > &cameras,
                    int ref_frame) const;

  /// warps image \in to the world volume at depth_slice,
  /// uses ni and nj as out's dimensions
  template<typename PixT>
  void warp_image_to_depth(const vil_image_view<PixT> &in,
                           vil_image_view<PixT> &out,
                           const vpgl_perspective_camera<double> &cam,
                           double depth_slice, int f, PixT fill);

  virtual vnl_double_3
  point_at_depth(unsigned int i, unsigned int j, double depth) const = 0;

  virtual vnl_double_3
  point_at_depth_on_axis(double i, double j, double depth) const = 0;

  unsigned int ni() const { return ni_; }
  unsigned int nj() const { return nj_; }

protected:
  unsigned int ni_, nj_;

  warp_image_parameters wip;
};

/// Return a subset of landmark points that project into the given region of interest
/// \param camera is the camera used to project the points
/// \param i0 is the horizontal coordinate of upper left corner of the ROI
/// \param ni is the width of the ROI
/// \param j0 is the vertical coordinate of upper left corner of the ROI
/// \param nj is the height of the ROI
/// \param landmarks is the set of 3D landmark points to project
/// \return the subset of \p landmarks that project into the ROI
std::vector<vnl_double_3>
filter_visible_landmarks(const vpgl_perspective_camera<double> &camera,
  int i0, int ni, int j0, int nj,
  const std::vector<vnl_double_3> &landmarks);

/// Robustly compute the bounding planes of the landmarks in a given direction
/// \param  landmarks is the set of 3D landmark points
/// \param  normal is the normal vector of the plane
/// \retval min_offset is the minimum plane offset
/// \retval max_offset is the maximum plane offset
/// \param  outlier_thresh is the threshold for fraction of outlier offsets to
///         reject at both the top and bottom
/// \param  safety_margin_factor is the fraction of total offset range to pad
///         both top and bottom to account for insufficient landmark samples
void
compute_offset_range(const std::vector<vnl_double_3> &landmarks,
  const vnl_vector_fixed<double, 3> &normal,
  double &min_offset, double &max_offset,
  const double outlier_thresh = 0.05,
  const double safety_margin_factor = 0.33);

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver


#endif
