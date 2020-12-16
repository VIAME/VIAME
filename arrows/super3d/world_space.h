// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
