// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of epipolar geometry functions.

#include "epipolar_geometry.h"
#include <viame/measurement/triangulate.h>

namespace viame {

namespace mvg {

/// Test corresponding points against a fundamental matrix and mark inliers
std::vector< bool >
mark_fm_inliers(
  viame::fundamental_matrix const& fm,
  std::vector< viame::vector_2d > const& pts1,
  std::vector< viame::vector_2d > const& pts2,
  double inlier_scale )
{
  using namespace viame;

  matrix_3x3d F = fm.matrix();
  matrix_3x3d Ft = F.transpose();

  std::vector< bool > inliers( std::min( pts1.size(), pts2.size() ) );
  for( size_t i = 0; i < inliers.size(); ++i )
  {
    const vector_2d& p1 = pts1[ i ];
    const vector_2d& p2 = pts2[ i ];
    vector_3d v1( p1.x(), p1.y(), 1.0 );
    vector_3d v2( p2.x(), p2.y(), 1.0 );
    vector_3d l1 = F * v1;
    vector_3d l2 = Ft * v2;
    double s1 = 1.0 / sqrt( l1.x() * l1.x() + l1.y() * l1.y() );
    double s2 = 1.0 / sqrt( l2.x() * l2.x() + l2.y() * l2.y() );
    // sum of point to epipolar line distance in both images
    double d = v1.dot( l2 ) * ( s1 + s2 );
    inliers[ i ] = std::fabs( d ) < inlier_scale;
  }
  return inliers;
}

/// Compute a valid left camera from an essential matrix
viame::simple_camera_perspective
extract_valid_left_camera(
  const viame::essential_matrix_d& e,
  const viame::vector_2d& left_pt,
  const viame::vector_2d& right_pt )
{
  using namespace viame;

  /// construct an identity right camera
  const vector_3d t = e.translation();
  rotation_d R = e.rotation();

  std::vector< vector_2d > pts;
  pts.push_back( right_pt );
  pts.push_back( left_pt );

  std::vector< viame::simple_camera_perspective > cams( 2 );
  const viame::simple_camera_perspective& left_camera = cams[ 1 ];

  // option 1
  cams[ 1 ] = viame::simple_camera_perspective( R.inverse() * -t, R );

  vector_3d pt3 = triangulate_inhomog( cams, pts );
  if( pt3.z() > 0.0 && left_camera.depth( pt3 ) > 0.0 )
  {
    return left_camera;
  }

  // option 2, with negated translation
  cams[ 1 ] = viame::simple_camera_perspective( R.inverse() * t, R );
  pt3 = triangulate_inhomog( cams, pts );
  if( pt3.z() > 0.0 && left_camera.depth( pt3 ) > 0.0 )
  {
    return left_camera;
  }

  // option 3, with the twisted pair rotation
  R = e.twisted_rotation();
  cams[ 1 ] = viame::simple_camera_perspective( R.inverse() * -t, R );
  pt3 = triangulate_inhomog( cams, pts );
  if( pt3.z() > 0.0 && left_camera.depth( pt3 ) > 0.0 )
  {
    return left_camera;
  }

  // option 4, with negated translation
  cams[ 1 ] = viame::simple_camera_perspective( R.inverse() * t, R );
  pt3 = triangulate_inhomog( cams, pts );
  if( pt3.z() > 0.0 && left_camera.depth( pt3 ) > 0.0 )
  {
    return left_camera;
  }
  // should never get here
  return viame::simple_camera_perspective();
}

// Compute the fundamental matrix from a pair of cameras
viame::fundamental_matrix_sptr
fundamental_matrix_from_cameras(
  viame::camera_perspective const& right_cam,
  viame::camera_perspective const& left_cam )
{
  using namespace viame;

  essential_matrix_sptr em = essential_matrix_from_cameras(
    right_cam,
    left_cam );
  return essential_matrix_to_fundamental(
    *em, *right_cam.intrinsics(),
    *left_cam.intrinsics() );
}

// Compute the essential matrix from a pair of cameras
viame::essential_matrix_sptr
essential_matrix_from_cameras(
  viame::camera_perspective const& right_cam,
  viame::camera_perspective const& left_cam )
{
  using namespace viame;

  rotation_d R1 = right_cam.rotation();
  rotation_d R2 = left_cam.rotation();
  vector_3d t1 = right_cam.translation();
  vector_3d t2 = left_cam.translation();
  rotation_d R( R2 * R1.inverse() );
  vector_3d t( t2 - R * t1 );
  return std::make_shared< essential_matrix_d >( R, t );
}

/// Convert an essential matrix to a fundamental matrix
viame::fundamental_matrix_sptr
essential_matrix_to_fundamental(
  viame::essential_matrix const& E,
  viame::camera_intrinsics const& right_cal,
  viame::camera_intrinsics const& left_cal )
{
  using namespace viame;

  matrix_3x3d Kr_inv = right_cal.as_matrix().inverse();
  matrix_3x3d Kl_invt = left_cal.as_matrix().transpose().inverse();
  return std::make_shared< fundamental_matrix_d >(
    Kl_invt * E.matrix() *
    Kr_inv );
}

} // namespace mvg

} // namespace viame
