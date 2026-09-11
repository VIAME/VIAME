/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief `library/measurement/projection` against what OpenCV's calib3d says
///
/// `tests/golden/projection/opencv.json` is written by
/// `record_from_opencv.py`, which calls `cv::projectPoints`,
/// `cv::undistortPoints`, `cv::stereoRectify` and
/// `cv::initUndistortRectifyMap` on four rigs: the golden measurement scene,
/// the camtrawl rig shipped in `examples/size_measurement`, a strongly
/// distorted wide-angle pair, and a **vertical** rig, which is the other
/// branch `stereo_rectify` takes.
///
/// Unlike the Eigen recording, this one can be regenerated: cv2 is a wheel
/// and stays after P7-T09. It is committed so the C++ test does not need
/// python, not because the numbers are unobtainable.

#include <viame/measurement/projection.h>

#include "../golden_json.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace kv = kwiver::vital;
namespace mp = viame::measurement;

using viame::testing::golden_json;

namespace {

// ----------------------------------------------------------------------------
std::string
golden_path()
{
#ifdef VIAME_GOLDEN_PROJECTION_DIR
  return std::string( VIAME_GOLDEN_PROJECTION_DIR ) + "/opencv.json";
#else
  return "opencv.json";
#endif
}

// ----------------------------------------------------------------------------
kv::matrix_3x3d
to_matrix( std::vector< double > const& values )
{
  kv::matrix_3x3d out;

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      out( r, c ) = values[ r * 3 + c ];
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
kv::matrix_3x4d
to_projection( std::vector< double > const& values )
{
  kv::matrix_3x4d out;

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 4; ++c )
    {
      out( r, c ) = values[ r * 4 + c ];
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Distortion, with an all-zero vector read back as "no distortion".
///
/// OpenCV treats a zero `distCoeffs` and a null one identically; this keeps
/// the same, and it matters because the zero path is the exact one.
mp::distortion_t
to_distortion( std::vector< double > const& values )
{
  for( auto const value : values )
  {
    if( value != 0.0 ) { return values; }
  }

  return {};
}

// A pixel position OpenCV computed in double and this computes in double:
// the two agree to the last few bits of the intermediate, not further,
// because the order of the multiplications differs.
constexpr double pixel_tolerance = 1e-9;

// The rectification's own tolerance, which is larger and says why in
// `stereo_rectify`: OpenCV samples the image border through `CV_32FC2`
// points, so its inscribed rectangle -- and the focal length scaled to fit
// it -- carry about seven significant digits. This is that, with room.
constexpr double rectification_relative = 2e-6;

// The maps are stored as float32 by OpenCV, so about seven digits on a
// coordinate of up to a couple of thousand.
constexpr double map_tolerance = 2e-3;

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( projection, project_point_matches_opencv )
{
  golden_json const g( golden_path() );
  auto const cases = g.section( "projection" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const name = golden_json::text( c, "rig" ) + " " +
                      golden_json::text( c, "side" );

    auto const intrinsics = to_matrix( golden_json::numbers( c, "intrinsics" ) );
    auto const distortion = to_distortion( golden_json::numbers( c, "distortion" ) );
    auto const points = golden_json::numbers( c, "points" );
    auto const expected = golden_json::numbers( c, "expected" );

    ASSERT_EQ( points.size() / 3, expected.size() / 2 ) << name;

    for( size_t i = 0; i * 3 < points.size(); ++i )
    {
      kv::vector_3d const point( points[ i * 3 ], points[ i * 3 + 1 ],
                                 points[ i * 3 + 2 ] );

      auto const got = mp::project_point( point, intrinsics, distortion );

      EXPECT_NEAR( got[ 0 ], expected[ i * 2 ], pixel_tolerance )
        << name << " point " << i << " x";
      EXPECT_NEAR( got[ 1 ], expected[ i * 2 + 1 ], pixel_tolerance )
        << name << " point " << i << " y";
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( projection, undistort_point_matches_opencv )
{
  golden_json const g( golden_path() );
  auto const cases = g.section( "undistortion" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const name = golden_json::text( c, "rig" ) + " " +
                      golden_json::text( c, "side" );

    auto const intrinsics = to_matrix( golden_json::numbers( c, "intrinsics" ) );
    auto const distortion = to_distortion( golden_json::numbers( c, "distortion" ) );
    auto const points = golden_json::numbers( c, "points" );
    auto const expected = golden_json::numbers( c, "expected" );

    ASSERT_EQ( points.size(), expected.size() ) << name;

    kv::matrix_3x4d widened;
    widened.setZero();
    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        widened( r, col ) = intrinsics( r, col );
      }
    }

    for( size_t i = 0; i * 2 < points.size(); ++i )
    {
      kv::vector_2d const point( points[ i * 2 ], points[ i * 2 + 1 ] );

      auto const got = mp::undistort_point(
        point, intrinsics, distortion, kv::matrix_3x3d::Identity(), widened );

      EXPECT_NEAR( got[ 0 ], expected[ i * 2 ], pixel_tolerance )
        << name << " point " << i << " x";
      EXPECT_NEAR( got[ 1 ], expected[ i * 2 + 1 ], pixel_tolerance )
        << name << " point " << i << " y";
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( projection, stereo_rectify_matches_opencv )
{
  golden_json const g( golden_path() );
  auto const cases = g.section( "rectification" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const name = golden_json::text( c, "rig" );

    auto const k_left = to_matrix( golden_json::numbers( c, "k_left" ) );
    auto const k_right = to_matrix( golden_json::numbers( c, "k_right" ) );
    auto const d_left = to_distortion( golden_json::numbers( c, "d_left" ) );
    auto const d_right = to_distortion( golden_json::numbers( c, "d_right" ) );
    auto const rotation = to_matrix( golden_json::numbers( c, "rotation" ) );
    auto const t = golden_json::numbers( c, "translation" );

    auto const width = static_cast< size_t >( golden_json::number( c, "width" ) );
    auto const height = static_cast< size_t >( golden_json::number( c, "height" ) );

    auto const got = mp::stereo_rectify(
      k_left, d_left, k_right, d_right, width, height, rotation,
      kv::vector_3d( t[ 0 ], t[ 1 ], t[ 2 ] ) );

    auto const r1 = golden_json::numbers( c, "r1" );
    auto const r2 = golden_json::numbers( c, "r2" );
    auto const p1 = golden_json::numbers( c, "p1" );
    auto const p2 = golden_json::numbers( c, "p2" );
    auto const q = golden_json::numbers( c, "q" );

    auto relative = [ & ]( double a, double b )
    {
      return rectification_relative * std::max( 1.0, std::abs( b ) );
    };

    for( unsigned r = 0; r < 3; ++r )
    {
      for( unsigned col = 0; col < 3; ++col )
      {
        EXPECT_NEAR( got.left_rotation( r, col ), r1[ r * 3 + col ],
                     relative( got.left_rotation( r, col ), r1[ r * 3 + col ] ) )
          << name << " R1 (" << r << ", " << col << ")";
        EXPECT_NEAR( got.right_rotation( r, col ), r2[ r * 3 + col ],
                     relative( got.right_rotation( r, col ), r2[ r * 3 + col ] ) )
          << name << " R2 (" << r << ", " << col << ")";
      }

      for( unsigned col = 0; col < 4; ++col )
      {
        EXPECT_NEAR( got.left_projection( r, col ), p1[ r * 4 + col ],
                     relative( got.left_projection( r, col ), p1[ r * 4 + col ] ) )
          << name << " P1 (" << r << ", " << col << ")";
        EXPECT_NEAR( got.right_projection( r, col ), p2[ r * 4 + col ],
                     relative( got.right_projection( r, col ), p2[ r * 4 + col ] ) )
          << name << " P2 (" << r << ", " << col << ")";
      }
    }

    for( unsigned r = 0; r < 4; ++r )
    {
      for( unsigned col = 0; col < 4; ++col )
      {
        EXPECT_NEAR( got.disparity_to_depth( r, col ), q[ r * 4 + col ],
                     relative( got.disparity_to_depth( r, col ),
                               q[ r * 4 + col ] ) )
          << name << " Q (" << r << ", " << col << ")";
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( projection, rectification_maps_match_opencv )
{
  golden_json const g( golden_path() );
  auto const cases = g.section( "rectification" );
  ASSERT_FALSE( cases.empty() );

  for( auto const& c : cases )
  {
    auto const name = golden_json::text( c, "rig" );

    auto const width = static_cast< size_t >( golden_json::number( c, "width" ) );
    auto const height = static_cast< size_t >( golden_json::number( c, "height" ) );

    for( unsigned side = 0; side < 2; ++side )
    {
      auto const key = side == 0 ? std::string( "left" ) : std::string( "right" );

      auto const intrinsics = to_matrix( golden_json::numbers( c, "k_" + key ) );
      auto const distortion = to_distortion( golden_json::numbers( c, "d_" + key ) );
      auto const rotation = to_matrix(
        golden_json::numbers( c, side == 0 ? "r1" : "r2" ) );
      auto const projection = to_projection(
        golden_json::numbers( c, side == 0 ? "p1" : "p2" ) );

      kv::image_of< float > map_x, map_y;
      mp::rectification_maps( intrinsics, distortion, rotation, projection,
                              width, height, map_x, map_y );

      auto const samples = golden_json::numbers( c, "map_" + key );
      ASSERT_EQ( samples.size() % 4u, 0u ) << name;

      for( size_t i = 0; i * 4 < samples.size(); ++i )
      {
        auto const x = static_cast< size_t >( samples[ i * 4 ] );
        auto const y = static_cast< size_t >( samples[ i * 4 + 1 ] );

        EXPECT_NEAR( map_x( x, y, 0 ), samples[ i * 4 + 2 ], map_tolerance )
          << name << " " << key << " map x at (" << x << ", " << y << ")";
        EXPECT_NEAR( map_y( x, y, 0 ), samples[ i * 4 + 3 ], map_tolerance )
          << name << " " << key << " map y at (" << x << ", " << y << ")";
      }
    }
  }
}
