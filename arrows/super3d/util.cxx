// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
 * \file
 * \brief Source file for util
 */

#include "util.h"

#include <vnl/vnl_inverse.h>

namespace kwiver {
namespace arrows {
namespace super3d {

/// Produce the camera corresponding to a downsampled image
/// \param camera The input camera
/// \param scale The scale of the downsampled image (default is 0.5)
/// \return A camera corresponding to the downsampled image
vpgl_perspective_camera<double>
scale_camera(const vpgl_perspective_camera<double>& camera, double scale)
{
  vpgl_perspective_camera<double> cam(camera);
  vpgl_calibration_matrix<double> K = cam.get_calibration();
  vgl_point_2d<double> pp = K.principal_point();
  pp.x() *= scale;
  pp.y() *= scale;
  K.set_principal_point(pp);
  K.set_focal_length(K.focal_length()*scale);
  cam.set_calibration(K);
  return cam;
}

//*****************************************************************************

/// Produce the camera corresponding to a cropped image
/// \param camera The input camera
/// \param left the left coordinate of the cropping
/// \param top the left coordinate of the cropping
/// \return A camera corresponding to the cropped image
vpgl_perspective_camera<double>
crop_camera(const vpgl_perspective_camera<double>& camera, double left, double top)
{
  vpgl_perspective_camera<double> cam(camera);
  vpgl_calibration_matrix<double> K = cam.get_calibration();
  vgl_point_2d<double> pp = K.principal_point();
  pp.x() -= left;
  pp.y() -= top;
  K.set_principal_point(pp);
  cam.set_calibration(K);
  return cam;
}

//*****************************************************************************

/// Convert a depth map into a height map
void depth_map_to_height_map(const vpgl_perspective_camera<double>& camera,
                             const vil_image_view<double>& depth_map,
                             vil_image_view<double>& height_map)
{
  const vnl_matrix_fixed<double, 3, 4>& P = camera.get_matrix();
  const vnl_vector_fixed<double, 3> v = vnl_inverse(P.extract(3, 3)).get_row(2);
  const double o = dot_product(v, -P.get_column(3));
  assert(depth_map.nplanes() == 1);
  height_map.set_size(depth_map.ni(), depth_map.nj(), 1);
  for (unsigned j = 0; j < depth_map.nj(); ++j)
  {
    for (unsigned i = 0; i < depth_map.ni(); ++i)
    {
      const double& d = depth_map(i, j);
      vnl_vector_fixed<double, 3> pt(i, j, 1);
      height_map(i, j) = d * dot_product(v, pt) + o;
    }
  }
}

//*****************************************************************************

/// Convert a height map into a depth map
void height_map_to_depth_map(const vpgl_perspective_camera<double>& camera,
                             const vil_image_view<double>& height_map,
                             vil_image_view<double>& depth_map)
{
  const vnl_matrix_fixed<double, 3, 4>& P = camera.get_matrix();
  const vnl_vector_fixed<double, 3> v = vnl_inverse(P.extract(3, 3)).get_row(2);
  const double o = dot_product(v, -P.get_column(3));
  assert(height_map.nplanes() == 1);
  depth_map.set_size(height_map.ni(), height_map.nj(), 1);
  for (unsigned j = 0; j < height_map.nj(); ++j)
  {
    for (unsigned i = 0; i < height_map.ni(); ++i)
    {
      const double& h = height_map(i, j);
      vnl_vector_fixed<double, 3> pt(i, j, 1);
      depth_map(i, j) = (h - o) / dot_product(v, pt);
    }
  }
}

//*****************************************************************************

/// Convert a height map into a depth map
void height_map_to_depth_map(vpgl_perspective_camera<double> const& camera,
                             vil_image_view<double> const& height_map,
                             vil_image_view<double>& depth_map,
                             vil_image_view<double>& uncertainty)
{
  const vnl_matrix_fixed<double, 3, 4>& P = camera.get_matrix();
  const vnl_vector_fixed<double, 3> v = vnl_inverse(P.extract(3, 3)).get_row(2);
  const double o = dot_product(v, -P.get_column(3));
  assert(height_map.nplanes() == 1);
  assert(uncertainty.nplanes() == 1);
  assert(uncertainty.ni() == height_map.ni());
  assert(uncertainty.nj() == height_map.nj());
  depth_map.set_size(height_map.ni(), height_map.nj(), 1);
  for (unsigned j = 0; j < height_map.nj(); ++j)
  {
    for (unsigned i = 0; i < height_map.ni(); ++i)
    {
      const double& h = height_map(i, j);
      vnl_vector_fixed<double, 3> pt(i, j, 1);
      const double s = 1.0 / dot_product(v, pt);
      depth_map(i, j) = (h - o) * s;
      uncertainty(i, j) *= std::abs(s);
    }
  }
}

//*****************************************************************************

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
