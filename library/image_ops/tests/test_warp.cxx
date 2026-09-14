/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// The geometric transforms, against what OpenCV computed for the same
/// pixels.
///
/// These cases were recorded against the window rather than the whole
/// fixture, because a resize of a window is not a window of a resize: the
/// only way to compare is to hand both implementations the same picture. So
/// the margins are zero and every pixel counts, edges included -- which is
/// where a sample grid that is half a pixel out shows up.

#include <image_ops/warp.h>

#include <viame/core_types/image.h>
#include <viame/core_types/matrix.h>

#include "golden_image.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

namespace golden_image = viame::testing::golden_image;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

/// The homography the recorder warped with.
kv::matrix_3x3d
recorded_homography()
{
  kv::matrix_3x3d h;
  h( 0, 0 ) =  1.10;  h( 0, 1 ) = 0.15;   h( 0, 2 ) = -3.0;
  h( 1, 0 ) = -0.08;  h( 1, 1 ) = 0.95;   h( 1, 2 ) =  2.0;
  h( 2, 0 ) =  0.0012; h( 2, 1 ) = 0.0008; h( 2, 2 ) = 1.0;
  return h;
}

void
check_resize( std::string const& name, io::interpolation how )
{
  auto const input = golden_image::input( name );

  golden_image::expect_matches(
    name,
    io::resize( input, golden_image::expected_width( name ),
                golden_image::expected_height( name ), how ) );
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( warp, resize_bilinear_matches_opencv )
{
  for( auto const* name : { "resize_bilinear_half", "resize_bilinear_double",
                            "resize_bilinear_odd", "resize_bilinear_rgb" } )
  {
    check_resize( name, io::interpolation::BILINEAR );
  }
}

// ----------------------------------------------------------------------------
TEST ( warp, resize_nearest_matches_opencv )
{
  for( auto const* name : { "resize_nearest_half", "resize_nearest_double",
                            "resize_nearest_odd" } )
  {
    check_resize( name, io::interpolation::NEAREST );
  }
}

// ----------------------------------------------------------------------------
TEST ( warp, resize_area_matches_opencv )
{
  for( auto const* name : { "resize_area_half", "resize_area_double",
                            "resize_area_odd" } )
  {
    check_resize( name, io::interpolation::AREA );
  }
}

// ----------------------------------------------------------------------------
TEST ( warp, warp_perspective_matches_opencv )
{
  auto const input = golden_image::input( "warp_perspective" );

  golden_image::expect_matches(
    "warp_perspective",
    io::warp_perspective( input, recorded_homography(), input.width(),
                          input.height(), io::interpolation::BILINEAR,
                          io::border_mode::CONSTANT ) );

  golden_image::expect_matches(
    "warp_perspective_replicate",
    io::warp_perspective( input, recorded_homography(), input.width(),
                          input.height(), io::interpolation::BILINEAR,
                          io::border_mode::REPLICATE ) );

  golden_image::expect_matches(
    "warp_perspective_nearest",
    io::warp_perspective( input, recorded_homography(), input.width(),
                          input.height(), io::interpolation::NEAREST,
                          io::border_mode::CONSTANT ) );
}

// ----------------------------------------------------------------------------
TEST ( warp, warp_affine_matches_opencv )
{
  auto const input = golden_image::input( "warp_affine_rotate" );
  auto const rotation = io::rotation_matrix_2d( 15.5, 11.5, 20.0, 1.15 );

  golden_image::expect_matches(
    "warp_affine_rotate",
    io::warp_affine( input, rotation, input.width(), input.height(),
                     io::interpolation::BILINEAR,
                     io::border_mode::CONSTANT ) );
}

// ----------------------------------------------------------------------------
TEST ( warp, remap_matches_opencv )
{
  auto const input = golden_image::input( "remap_wave" );

  kv::image_of< float > map_x( input.width(), input.height(), 1 );
  kv::image_of< float > map_y( input.width(), input.height(), 1 );

  for( size_t j = 0; j < input.height(); ++j )
  {
    for( size_t i = 0; i < input.width(); ++i )
    {
      map_x( i, j, 0 ) = static_cast< float >(
        static_cast< double >( i ) +
        3.0 * std::sin( static_cast< double >( j ) / 4.0 ) );
      map_y( i, j, 0 ) = static_cast< float >(
        static_cast< double >( j ) +
        2.0 * std::cos( static_cast< double >( i ) / 5.0 ) );
    }
  }

  golden_image::expect_matches(
    "remap_wave",
    io::remap( input, map_x, map_y, io::interpolation::BILINEAR,
               io::border_mode::CONSTANT ) );
}

// ----------------------------------------------------------------------------
/// The bilinear warps agree with OpenCV to about four counts rather than
/// exactly, and the recording's tolerance says so. This says whose four
/// counts they are.
///
/// OpenCV interpolates a warp in fixed point -- INTER_BITS is 5, so the
/// position is quantised to a thirty-second of a pixel -- and `image_ops`
/// interpolates in double. Both are approximations of the same exact
/// bilinear answer, and the claim is that this one is nearer it. Computing
/// that answer at full precision and measuring both against it is the only
/// way to tell "more accurate" from "differently wrong".
TEST ( warp, is_nearer_the_exact_answer_than_opencv_is )
{
  auto const input = golden_image::input( "warp_perspective" );
  auto const homography = recorded_homography();
  auto const inverse = homography.inverse();

  auto const mine =
    io::warp_perspective( input, homography, input.width(), input.height(),
                          io::interpolation::BILINEAR,
                          io::border_mode::CONSTANT );

  auto const opencv = golden_image::read< double >(
    golden_image::find( "warp_perspective" ), "expected" );

  double mine_error = 0.0;
  double opencv_error = 0.0;
  size_t counted = 0;

  for( size_t j = 0; j < input.height(); ++j )
  {
    for( size_t i = 0; i < input.width(); ++i )
    {
      auto const dx = static_cast< double >( i );
      auto const dy = static_cast< double >( j );
      auto const w =
        inverse( 2, 0 ) * dx + inverse( 2, 1 ) * dy + inverse( 2, 2 );

      if( w == 0.0 ) { continue; }

      auto const sx =
        ( inverse( 0, 0 ) * dx + inverse( 0, 1 ) * dy + inverse( 0, 2 ) ) / w;
      auto const sy =
        ( inverse( 1, 0 ) * dx + inverse( 1, 1 ) * dy + inverse( 1, 2 ) ) / w;

      // Only where the whole four-pixel footprint is inside the image, so
      // that the border rule is not what is being measured
      if( sx < 0.0 || sy < 0.0 ||
          sx >= static_cast< double >( input.width() ) - 1.0 ||
          sy >= static_cast< double >( input.height() ) - 1.0 )
      {
        continue;
      }

      auto const exact = io::sample_bilinear( input, sx, sy, 0,
                                              io::border_mode::CONSTANT );

      mine_error += std::abs( static_cast< double >( mine( i, j, 0 ) ) -
                              exact );
      opencv_error += std::abs( opencv( i, j, 0 ) - exact );
      ++counted;
    }
  }

  ASSERT_GT( counted, 100u ) << "too few interior pixels to say anything";

  auto const mine_mean = mine_error / static_cast< double >( counted );
  auto const opencv_mean = opencv_error / static_cast< double >( counted );

  std::cout << "[          ] warp against the exact answer: image_ops "
            << mine_mean << ", OpenCV " << opencv_mean << " over " << counted
            << " interior pixels" << std::endl;

  // Rounding to a byte costs a quarter of a count on average, so this is
  // about as near as an integer result can be
  EXPECT_LT( mine_mean, 0.3 );
  EXPECT_LT( mine_mean, opencv_mean );
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give.
TEST ( warp, resize_to_the_same_size_changes_nothing )
{
  kv::image_of< uint8_t > image( 5, 4, 2 );
  uint8_t value = 3;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t p = 0; p < image.depth(); ++p )
      {
        image( i, j, p ) = static_cast< uint8_t >( value = value * 7 + 1 );
      }
    }
  }

  for( auto const how : { io::interpolation::NEAREST,
                          io::interpolation::BILINEAR,
                          io::interpolation::AREA } )
  {
    auto const same = io::resize( image, 5, 4, how );

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        for( size_t p = 0; p < image.depth(); ++p )
        {
          EXPECT_EQ( image( i, j, p ), same( i, j, p ) )
            << "mode " << int( how ) << " at (" << i << ", " << j << ")";
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// Halving a block-constant image is exact under every mode, which is the
/// one case where the three have to agree.
TEST ( warp, halving_a_block_image_is_exact )
{
  kv::image_of< uint8_t > blocks( 8, 6, 1 );

  for( size_t j = 0; j < blocks.height(); ++j )
  {
    for( size_t i = 0; i < blocks.width(); ++i )
    {
      blocks( i, j, 0 ) =
        static_cast< uint8_t >( 20 * ( ( j / 2 ) * 4 + ( i / 2 ) ) );
    }
  }

  for( auto const how : { io::interpolation::NEAREST,
                          io::interpolation::BILINEAR,
                          io::interpolation::AREA } )
  {
    auto const half = io::resize( blocks, 4, 3, how );

    for( size_t j = 0; j < half.height(); ++j )
    {
      for( size_t i = 0; i < half.width(); ++i )
      {
        EXPECT_EQ( 20 * ( j * 4 + i ), half( i, j, 0 ) )
          << "mode " << int( how ) << " at (" << i << ", " << j << ")";
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// The identity homography leaves the image alone.
TEST ( warp, an_identity_warp_changes_nothing )
{
  kv::image_of< uint8_t > image( 6, 4, 1 );
  uint8_t value = 11;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( value = value * 5 + 3 );
    }
  }

  auto const same = io::warp_perspective( image, kv::matrix_3x3d::Identity() );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      EXPECT_EQ( image( i, j, 0 ), same( i, j, 0 ) );
    }
  }
}

// ----------------------------------------------------------------------------
/// A translation by whole pixels moves the picture and nothing else.
TEST ( warp, a_whole_pixel_translation_is_a_shift )
{
  kv::image_of< uint8_t > image( 6, 4, 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( 10 * i + j );
    }
  }

  kv::matrix_3x3d shift = kv::matrix_3x3d::Identity();
  shift( 0, 2 ) = 2.0;   // two to the right
  shift( 1, 2 ) = 1.0;   // one down

  auto const moved = io::warp_perspective( image, shift );

  for( size_t j = 1; j < image.height(); ++j )
  {
    for( size_t i = 2; i < image.width(); ++i )
    {
      EXPECT_EQ( image( i - 2, j - 1, 0 ), moved( i, j, 0 ) )
        << "at (" << i << ", " << j << ")";
    }
  }

  // and the vacated strip is the constant border
  EXPECT_EQ( 0, moved( 0, 0, 0 ) );
  EXPECT_EQ( 0, moved( 1, 3, 0 ) );
}

// ----------------------------------------------------------------------------
/// The rotation matrix rotates about the centre it is given.
TEST ( warp, rotation_matrix_fixes_its_centre )
{
  auto const rotation = io::rotation_matrix_2d( 4.0, 3.0, 37.0, 1.0 );

  auto const x = rotation( 0, 0 ) * 4.0 + rotation( 0, 1 ) * 3.0 +
                 rotation( 0, 2 );
  auto const y = rotation( 1, 0 ) * 4.0 + rotation( 1, 1 ) * 3.0 +
                 rotation( 1, 2 );

  EXPECT_NEAR( 4.0, x, 1e-12 );
  EXPECT_NEAR( 3.0, y, 1e-12 );
}

// ----------------------------------------------------------------------------
/// An identity remap is a copy, and a shifted one is a shift.
TEST ( warp, remap_by_identity_is_a_copy )
{
  kv::image_of< uint8_t > image( 5, 4, 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( 7 * i + 3 * j );
    }
  }

  kv::image_of< float > map_x( 5, 4, 1 );
  kv::image_of< float > map_y( 5, 4, 1 );

  for( size_t j = 0; j < 4; ++j )
  {
    for( size_t i = 0; i < 5; ++i )
    {
      map_x( i, j, 0 ) = static_cast< float >( i );
      map_y( i, j, 0 ) = static_cast< float >( j );
    }
  }

  auto const same = io::remap( image, map_x, map_y );

  for( size_t j = 0; j < 4; ++j )
  {
    for( size_t i = 0; i < 5; ++i )
    {
      EXPECT_EQ( image( i, j, 0 ), same( i, j, 0 ) );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( warp, refuses_what_it_cannot_do )
{
  kv::image_of< uint8_t > image( 4, 4, 1 );

  EXPECT_THROW( io::resize( image, 0, 4 ), std::invalid_argument );
  EXPECT_THROW( io::resize( image, 4, 0 ), std::invalid_argument );

  // A singular homography has no inverse to warp by
  kv::matrix_3x3d singular = kv::matrix_3x3d::Zero();
  EXPECT_THROW( io::warp_perspective( image, singular ),
                std::invalid_argument );

  kv::image_of< float > map_x( 4, 4, 1 );
  kv::image_of< float > wrong( 3, 4, 1 );
  EXPECT_THROW( io::remap( image, map_x, wrong ), std::invalid_argument );

  kv::image_of< float > deep( 4, 4, 2 );
  EXPECT_THROW( io::remap( image, deep, map_x ), std::invalid_argument );
}
