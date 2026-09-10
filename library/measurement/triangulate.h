// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for MVG triangulation function

#ifndef KWIVER_ARROWS_MVG_TRIANGULATE_H_
#define KWIVER_ARROWS_MVG_TRIANGULATE_H_

#include "viame_measurement_export.h"
#include <viame/algorithm_framework/vital_config.h>

#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/camera_rpc.h>

namespace kwiver {

namespace arrows {

namespace mvg {

/// Triangulate a 3D point from a set of cameras and 2D image points
///
///  This function computes a linear least squares solution find a 3D point
///  that is the closest intersection of all the rays using an inhomogeneous
///  system of equations.  This method is affine invariant but does not work
///  for 3D points at infinity.
///
///  \param cameras a vector of camera objects
///  \param points a vector of image points corresponding to each camera
///  \return a 3D triangulated point location
template < typename T >
VIAME_MEASUREMENT_EXPORT
vital::vector_< 3, T >
triangulate_inhomog(
  const std::vector< vital::simple_camera_perspective >& cameras,
  const std::vector< vital::vector_< 2, T > >& points );

/// Triangulate a 3D point from a set of cameras and 2D image points
///
/// This function uses only the first two cameras and two points to triangulate.
/// It uses the method laid out in the paper "Triangulation Made Easy" Lindstrom
/// CVPR 2010.  This approach is efficient and offers an alternative approach
/// that
/// may be numerically superior in some cases.  This approach does not work for
/// points at infinty.
///
/// \param camera0 the first camera
/// \param camera1 the second camera
/// \param point0 a 2d point in the first camera
/// \param point1 a matching 2d point in the second camera
/// \return a 3D triangulated point location
template < typename T >
VIAME_MEASUREMENT_EXPORT
vital::vector_< 3, T >
triangulate_fast_two_view(
  const vital::simple_camera_perspective& camera0,
  const vital::simple_camera_perspective& camera1,
  const vital::vector_< 2, T >& point0,
  const vital::vector_< 2, T >& point1 );

/// Triangulate a homogeneous 3D point from a set of cameras and 2D image points
///
///  This function computes a linear least squares solution find a homogeneous
///  3D point that is the closest intersection of all the rays using a
///  homogeneous system of equations.  This method is not invariant to
///  tranformations but does allow for 3D points at infinity.
///
///  \param cameras a vector of camera objects
///  \param points a vector of image points corresponding to each camera
///  \return a homogeneous 3D triangulated point location
template < typename T >
VIAME_MEASUREMENT_EXPORT
vital::vector_< 4, T >
triangulate_homog(
  const std::vector< vital::simple_camera_perspective >& cameras,
  const std::vector< vital::vector_< 2, T > >& points );

/// Triangulate a 3D point from a set of RPC cameras and 2D image points
///
///  This function constructs rays at two arbitary heights using the cameras and
///  image points. Then a least squares solution is used to find the 3D point
///
///  \param cameras a vector of RPC camera objects
///  \param points a vector of image points corresponding to each camera
///  \return a 3D triangulated point location
template < typename T >
VIAME_MEASUREMENT_EXPORT
vital::vector_< 3, T >
triangulate_rpc(
  const std::vector< vital::simple_camera_rpc >& cameras,
  const std::vector< vital::vector_< 2, T > >& points );

} // end namespace mvg

} // end namespace arrows

} // end namespace kwiver

#endif
