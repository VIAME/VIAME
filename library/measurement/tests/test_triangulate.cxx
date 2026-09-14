/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Stereo triangulation and the camera matrices it is built on
///
/// Written because none of it was covered and all of it was **broken**.
/// P6 replaced Eigen with `core_types/math`, where `block()` and `row()`
/// return values rather than the writable proxies Eigen returns, so the four
/// `P.block< 3, 3 >( 0, 0 ) = R;` lines in `camera_perspective.cxx` and the
/// four `design_matrix.row( n ) = ...` lines here compiled, assigned to a
/// temporary and did nothing. `pose_matrix()` returned a zero matrix and
/// every stereo measurement VIAME computed came out zero. Finding 1.20.
///
/// The numbers are checked against geometry rather than against a recording:
/// a point put at a known place in front of a known rig projects to two
/// known pixels, and triangulating those two pixels has to give the point
/// back. That is a contract no refactoring can quietly weaken.

#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/similarity.h>
#include <viame/measurement/triangulate.h>

#include <gtest/gtest.h>

#include <cmath>

using namespace kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// A rotation matrix from an axis-angle vector.
matrix_3x3d
rodrigues( double x, double y, double z )
{
  double const angle = std::sqrt( x * x + y * y + z * z );

  if( angle == 0.0 )
  {
    return matrix_3x3d::Identity();
  }

  double const ax = x / angle, ay = y / angle, az = z / angle;

  matrix_3x3d cross;
  cross << 0.0, -az, ay,
           az, 0.0, -ax,
           -ay, ax, 0.0;

  return matrix_3x3d::Identity() + std::sin( angle ) * cross +
         ( 1.0 - std::cos( angle ) ) * ( cross * cross );
}

// ----------------------------------------------------------------------------
/// The rig `tests/golden/measurement`'s synthetic views are rendered through.
struct stereo_rig
{
  stereo_rig()
  {
    matrix_3x3d k_left;
    k_left << 600.0, 0.0, 319.5,
              0.0, 600.0, 239.5,
              0.0, 0.0, 1.0;

    matrix_3x3d k_right;
    k_right << 610.0, 0.0, 319.5,
               0.0, 610.0, 239.5,
               0.0, 0.0, 1.0;

    rotation = rodrigues( 0.004, -0.02, 0.001 );
    translation = vector_3d( -120.0, 2.0, 5.0 );

    matrix_3x3d const identity = matrix_3x3d::Identity();

    left = simple_camera_perspective(
      vector_3d( 0.0, 0.0, 0.0 ), rotation_d( identity ),
      std::make_shared< simple_camera_intrinsics >( k_left ) );

    right = simple_camera_perspective(
      vector_3d( -rotation.transpose() * translation ),
      rotation_d( rotation ),
      std::make_shared< simple_camera_intrinsics >( k_right ) );
  }

  matrix_3x3d rotation;
  vector_3d translation;
  simple_camera_perspective left;
  simple_camera_perspective right;
};

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( triangulate, pose_matrix_is_rotation_beside_translation )
{
  stereo_rig const rig;

  auto const pose = rig.right.pose_matrix();

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      EXPECT_NEAR( pose( r, c ), rig.rotation( r, c ), 1e-12 )
        << "rotation block at (" << r << ", " << c << ")";
    }

    EXPECT_NEAR( pose( r, 3 ), rig.translation[ r ], 1e-9 )
      << "translation column at row " << r;
  }
}

// ----------------------------------------------------------------------------
TEST ( triangulate, pose_matrix_of_the_identity_camera_is_identity )
{
  stereo_rig const rig;

  auto const pose = rig.left.pose_matrix();

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 4; ++c )
    {
      EXPECT_NEAR( pose( r, c ), r == c ? 1.0 : 0.0, 1e-12 );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( triangulate, as_matrix_is_the_intrinsics_times_the_pose )
{
  stereo_rig const rig;

  auto const projection = rig.right.as_matrix();
  auto const intrinsics = rig.right.intrinsics()->as_matrix();
  auto const pose = rig.right.pose_matrix();

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 4; ++c )
    {
      double expected = 0.0;
      for( unsigned k = 0; k < 3; ++k )
      {
        expected += intrinsics( r, k ) * pose( k, c );
      }

      EXPECT_NEAR( projection( r, c ), expected, 1e-9 );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( triangulate, a_projected_point_triangulates_back_to_itself )
{
  stereo_rig const rig;

  // Points spread in depth and across the frame, all in front of both
  // cameras and all well away from the epipole.
  std::vector< vector_3d > const points = {
    vector_3d( 0.0, 0.0, 2000.0 ),
    vector_3d( -400.0, -250.0, 2200.0 ),
    vector_3d( 350.0, 180.0, 1500.0 ),
    vector_3d( 120.0, -300.0, 4400.0 ),
    vector_3d( -60.0, 40.0, 900.0 ),
  };

  for( auto const& point : points )
  {
    auto const in_left = rig.left.project( point );
    auto const in_right = rig.right.project( point );

    auto const found = kwiver::arrows::mvg::triangulate_fast_two_view(
      rig.left, rig.right,
      vector_< 2, double >( in_left[ 0 ], in_left[ 1 ] ),
      vector_< 2, double >( in_right[ 0 ], in_right[ 1 ] ) );

    for( unsigned i = 0; i < 3; ++i )
    {
      // A thousandth of the range. Not tighter, because
      // `find_optimal_image_points` corrects the two rays with a first order
      // step rather than exactly, which costs about two parts in ten
      // thousand; the reference build on `main` gives the same answer to
      // thirteen digits, checked, so this is the algorithm rather than the
      // arithmetic. The contract is that a projected point comes back, and
      // returning zero misses it by the whole range.
      EXPECT_NEAR( found[ i ], point[ i ], 1e-3 * point[ 2 ] )
        << "component " << i << " of a point at depth " << point[ 2 ];
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( triangulate, a_known_segment_measures_its_own_length )
{
  stereo_rig const rig;

  vector_3d const head( -350.0, -120.0, 2100.0 );
  vector_3d const tail( -50.0, 80.0, 2180.0 );

  auto const found_head = kwiver::arrows::mvg::triangulate_fast_two_view(
    rig.left, rig.right,
    vector_< 2, double >( rig.left.project( head )[ 0 ],
                          rig.left.project( head )[ 1 ] ),
    vector_< 2, double >( rig.right.project( head )[ 0 ],
                          rig.right.project( head )[ 1 ] ) );

  auto const found_tail = kwiver::arrows::mvg::triangulate_fast_two_view(
    rig.left, rig.right,
    vector_< 2, double >( rig.left.project( tail )[ 0 ],
                          rig.left.project( tail )[ 1 ] ),
    vector_< 2, double >( rig.right.project( tail )[ 0 ],
                          rig.right.project( tail )[ 1 ] ) );

  double const expected = ( tail - head ).norm();
  double const measured =
    ( vector_3d( found_tail[ 0 ], found_tail[ 1 ], found_tail[ 2 ] ) -
      vector_3d( found_head[ 0 ], found_head[ 1 ], found_head[ 2 ] ) ).norm();

  EXPECT_GT( expected, 0.0 );
  EXPECT_NEAR( measured, expected, 1e-3 * expected );
}

// ----------------------------------------------------------------------------
/// `similarity_::matrix()` filled its corner and nothing else, for the same
/// reason. It is here rather than in the math tests because the defect is the
/// one this file exists for.
TEST ( triangulate, a_similarity_transform_becomes_its_own_matrix )
{
  similarity_d const transform(
    2.5, rotation_d( rodrigues( 0.1, -0.2, 0.3 ) ),
    vector_3d( 11.0, -22.0, 33.0 ) );

  auto const mat = transform.matrix();
  auto const rotation = transform.rotation().matrix();

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      EXPECT_NEAR( mat( r, c ), 2.5 * rotation( r, c ), 1e-12 );
    }

    EXPECT_NEAR( mat( r, 3 ), transform.translation()[ r ], 1e-12 );
    EXPECT_NEAR( mat( 3, r ), 0.0, 1e-12 );
  }

  EXPECT_NEAR( mat( 3, 3 ), 1.0, 1e-12 );

  // And it round trips, which is what every caller of it relies on
  similarity_d const back( mat );

  EXPECT_NEAR( back.scale(), transform.scale(), 1e-12 );

  for( unsigned i = 0; i < 3; ++i )
  {
    EXPECT_NEAR( back.translation()[ i ], transform.translation()[ i ], 1e-12 );
  }
}
