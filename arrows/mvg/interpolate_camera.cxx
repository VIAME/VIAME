// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of camera interpolation functions
 */

#include "interpolate_camera.h"

namespace kwiver {
namespace arrows {
namespace mvg {

/// Generate an interpolated camera between \c A and \c B by a given fraction \c f
vital::simple_camera_perspective
interpolate_camera(vital::simple_camera_perspective const& A,
                   vital::simple_camera_perspective const& B, double f)
{
  const double f1 = 1.0 - f;

  // interpolate center
  vital::vector_3d c = f1*A.get_center() + f*B.get_center();

  // interpolate rotation
  vital::rotation_d R = interpolate_rotation(A.get_rotation(), B.get_rotation(), f);

  // interpolate intrinsics
  vital::camera_intrinsics_sptr k1 = A.get_intrinsics(),
                                k2 = B.get_intrinsics();

  if( k1 == k2 )
  {
    return vital::simple_camera_perspective(c, R, k1);
  }

  double focal_len = f1*k1->focal_length() + f*k2->focal_length();
  vital::vector_2d principal_point = f1*k1->principal_point() + f*k2->principal_point();
  double aspect_ratio = f1*k1->aspect_ratio() + f*k2->aspect_ratio();
  double skew = f1*k1->skew() + f*k2->skew();
  vital::simple_camera_intrinsics k(focal_len, principal_point, aspect_ratio, skew);
  return vital::simple_camera_perspective(c, R, k);
}

/// Generate N evenly interpolated cameras in between \c A and \c B
void
interpolated_cameras(vital::simple_camera_perspective const& A,
                     vital::simple_camera_perspective const& B,
                     size_t n,
                     std::vector< vital::simple_camera_perspective > & interp_cams)
{
  interp_cams.reserve(interp_cams.capacity() + n);
  size_t denom = n + 1;
  for (size_t i=1; i < denom; ++i)
  {
    interp_cams.push_back(interpolate_camera(A, B, static_cast<double>(i) / denom));
  }
}

/// Genreate an interpolated camera from sptrs
vital::camera_perspective_sptr
interpolate_camera(vital::camera_perspective_sptr A,
                   vital::camera_perspective_sptr B, double f)
{
  if( A == B )
  {
    return A;
  }
  return std::dynamic_pointer_cast<vital::camera_perspective>(
    interpolate_camera(vital::simple_camera_perspective(*A),
    vital::simple_camera_perspective(*B), f).clone());
}

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver
