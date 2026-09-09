/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// A/B the in-house homography overlap against the VXL one it replaces.
///
/// While VXL is still built this compares the two directly over a spread of
/// homographies, which is far stronger evidence than expectations written by
/// hand. When VXL goes the comparison is dropped and the remaining tests,
/// which check values that are known analytically, stay.

#include <image_ops/homography_overlap.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#ifdef VIAME_TEST_HAVE_VXL
#include <arrows/vxl/compute_homography_overlap.h>
#include <vnl/vnl_double_3x3.h>
#endif

using namespace viame::image_ops;

namespace {

// ----------------------------------------------------------------------------
/// A row major 3x3, as the in-house routine takes it.
typedef std::vector< double > matrix;

matrix
identity()
{
  return { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
}

matrix
translation( double dx, double dy )
{
  return { 1, 0, dx, 0, 1, dy, 0, 0, 1 };
}

matrix
scaling( double sx, double sy )
{
  return { sx, 0, 0, 0, sy, 0, 0, 0, 1 };
}

matrix
rotation( double radians )
{
  auto const c = std::cos( radians );
  auto const s = std::sin( radians );
  return { c, -s, 0, s, c, 0, 0, 0, 1 };
}

// ----------------------------------------------------------------------------
/// Homographies worth checking: the identity, pure shifts on and off frame,
/// rotations, scales either way, and a projective one.
std::vector< std::pair< std::string, matrix > >
cases()
{
  std::vector< std::pair< std::string, matrix > > result = {
    { "identity", identity() },
    { "shift_quarter", translation( 160, 0 ) },
    { "shift_half", translation( 320, 240 ) },
    { "shift_off_frame", translation( 1000, 1000 ) },
    { "shift_negative", translation( -160, -120 ) },
    { "rotate_small", rotation( 0.05 ) },
    { "rotate_quarter_turn", rotation( 3.14159265358979 / 2.0 ) },
    { "scale_up", scaling( 2.0, 2.0 ) },
    { "scale_down", scaling( 0.5, 0.5 ) },
    { "scale_anisotropic", scaling( 1.5, 0.6 ) },
    { "projective", { 1.0, 0.1, 5.0, 0.05, 1.0, -3.0, 1e-4, 5e-5, 1.0 } },
    { "degenerate", { 0, 0, 0, 0, 0, 0, 0, 0, 0 } },
  };

  // A spread of random near-identity homographies, which is what a stabiliser
  // actually feeds this
  std::mt19937 engine( 20260909 );
  std::uniform_real_distribution< double > jitter( -0.25, 0.25 );

  for( int index = 0; index < 40; ++index )
  {
    result.emplace_back(
      "random_" + std::to_string( index ),
      matrix{ 1.0 + jitter( engine ), jitter( engine ),
              jitter( engine ) * 400.0,
              jitter( engine ), 1.0 + jitter( engine ),
              jitter( engine ) * 400.0,
              jitter( engine ) * 1e-4, jitter( engine ) * 1e-4, 1.0 } );
  }

  return result;
}

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, the_identity_covers_the_whole_frame )
{
  EXPECT_DOUBLE_EQ( 1.0, homography_overlap( identity().data(), 640, 480 ) );
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, an_empty_frame_has_no_overlap )
{
  EXPECT_DOUBLE_EQ( 0.0, homography_overlap( identity().data(), 0, 480 ) );
  EXPECT_DOUBLE_EQ( 0.0, homography_overlap( identity().data(), 640, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, a_shift_leaves_the_expected_fraction )
{
  // Shifting a 640 x 480 frame by a quarter of its width leaves three
  // quarters of it on screen
  auto const shifted = translation( 160, 0 );
  EXPECT_NEAR( 0.75, homography_overlap( shifted.data(), 640, 480 ), 1e-9 );

  // Half the width and half the height leaves a quarter
  auto const diagonal = translation( 320, 240 );
  EXPECT_NEAR( 0.25, homography_overlap( diagonal.data(), 640, 480 ), 1e-9 );
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, a_shift_clear_of_the_frame_has_no_overlap )
{
  auto const gone = translation( 1000, 1000 );
  EXPECT_DOUBLE_EQ( 0.0, homography_overlap( gone.data(), 640, 480 ) );
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, halving_the_scale_leaves_a_quarter )
{
  // The warped frame is half as wide and half as tall, and sits in the
  // corner of the original, so a quarter of the area is shared
  auto const smaller = scaling( 0.5, 0.5 );
  EXPECT_NEAR( 0.25, homography_overlap( smaller.data(), 640, 480 ), 1e-9 );
}

// ----------------------------------------------------------------------------
TEST ( homography_overlap, a_degenerate_homography_has_no_overlap )
{
  matrix const zero( 9, 0.0 );
  EXPECT_DOUBLE_EQ( 0.0, homography_overlap( zero.data(), 640, 480 ) );
}

#ifdef VIAME_TEST_HAVE_VXL

// ----------------------------------------------------------------------------
TEST ( homography_overlap, matches_the_vxl_implementation )
{
  for( auto const& entry : cases() )
  {
    vnl_double_3x3 h;
    for( unsigned row = 0; row < 3; ++row )
    {
      for( unsigned column = 0; column < 3; ++column )
      {
        h( row, column ) = entry.second[ row * 3 + column ];
      }
    }

    auto const expected =
      kwiver::arrows::vxl::overlap( h, 640, 480 );
    auto const actual =
      homography_overlap( entry.second.data(), 640, 480 );

    EXPECT_NEAR( expected, actual, 1e-9 ) << "case " << entry.first;
  }
}

#endif
