// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
 * \file
 * \brief Header file for world_angled_frustum, world space with an angled frustum that
 *        aligned with a normal (often the ground plane's)
 */
#ifndef KWIVER_ARROWS_SUPER3D_WORLD_ANGLED_FRUSTUM_H_
#define KWIVER_ARROWS_SUPER3D_WORLD_ANGLED_FRUSTUM_H_

#include "world_space.h"

namespace kwiver {
namespace arrows {
namespace super3d {

/// A world represetation using camera frustum and angled world planes
///
/// This world space is defined by the frustum of a refrence image, but
/// the depth slices are define by a planes in the world space at any
/// orientation.  This may result in a skewed volume.
class world_angled_frustum : public world_space
{
public:

  /// Constructor
  /// \param plane_normal is the normal of the planes in world space
  /// \param min_offset is the offset from the origin of the minimum plane
  /// \param max_offset is the offset from the origin of the maximum plane
  /// \param image_width width of the input image
  /// \param image_height height of the input image
  world_angled_frustum(const vpgl_perspective_camera<double> &cam,
                       const vnl_vector_fixed<double,3>& plane_normal,
                       double min_offset,
                       double max_offset,
                       unsigned int image_width,
                       unsigned int image_height);

  /// returns the corner points of an image slice at depth slice.
  /// depth slice is a value between 0 and 1 over the depth range
  std::vector<vnl_double_3> get_slice(double depth_slice) const;

  /// Maps a slice value at image location (i,j) to a depth value
  /// \param i the horizontal image coordinate
  /// \param j the vertical image coordinate
  /// \param slice is the slice value in the range [0,1]
  double slice_to_depth(double i, double j, double slice) const;

  /// Maps a slice value at image location (i,j) to a 3D world point
  /// \param i the horizontal image coordinate
  /// \param j the vertical image coordinate
  /// \param slice is the slice value in the range [0,1]
  vnl_double_3 point_at_depth_on_axis(double i, double j, double slice) const;

  vnl_double_3 point_at_depth(unsigned int i, unsigned int j, double slice) const
  {
    return point_at_depth_on_axis(i, j, slice);
  }

private:

  /// the offset of the minimum plane (slice = 0)
  double min_offset_;
  /// the seperation between the minimum and maximum planes
  double offset_range_;
  /// the normal of the world planes defining the slices
  vnl_double_3 normal_;
  /// the reference camera
  vpgl_perspective_camera<double> ref_cam_;

  /// cached value of inv(K*R)
  vnl_double_3x3 KR_inv_;
  /// camera center as a vnl_vector_fixed
  vnl_double_3 cam_center_;
  /// cached height offset
  double height_offset_;
  /// cached vector used in the denominator of the depth equation
  vnl_double_3 denom_vec_;
};

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver

#endif
