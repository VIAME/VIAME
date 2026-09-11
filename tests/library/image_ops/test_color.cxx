/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// The colour conversions, against what OpenCV computed for the same pixels.
///
/// `tests/golden/image_ops/opencv.json` was recorded by
/// `record_from_opencv.py` while OpenCV was still on the path. Reading a file
/// rather than calling OpenCV is what lets this test outlive it, which is the
/// whole point: after phase 7 there is no second implementation to compare
/// against, so the comparison has to have been made and written down.
///
/// The tolerances are per case and come from the recording, so raising one is
/// a change to a committed file rather than to a number in a test.

#include <image_ops/color.h>

#include <viame/core_types/image.h>

#include "golden_image.h"

#include <gtest/gtest.h>

#include <cstdint>
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

// ----------------------------------------------------------------------------
/// The recorded input is the crop, so a per-pixel conversion needs no more.
TEST ( color, rgb_to_gray_matches_opencv )
{
  golden_image::expect_matches(
    "rgb_to_gray", io::rgb_to_gray( golden_image::input( "rgb_to_gray" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, gray_to_rgb_matches_opencv )
{
  golden_image::expect_matches(
    "gray_to_rgb", io::gray_to_rgb( golden_image::input( "gray_to_rgb" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, swap_rb_matches_opencv )
{
  golden_image::expect_matches(
    "swap_rb", io::swap_rb( golden_image::input( "swap_rb" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, rgb_to_hsv_matches_opencv )
{
  golden_image::expect_matches(
    "rgb_to_hsv", io::rgb_to_hsv( golden_image::input( "rgb_to_hsv" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, rgb_to_lab_matches_opencv )
{
  golden_image::expect_matches(
    "rgb_to_lab", io::rgb_to_lab( golden_image::input( "rgb_to_lab" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, hsv_to_rgb_matches_opencv )
{
  golden_image::expect_matches(
    "hsv_to_rgb", io::hsv_to_rgb( golden_image::input( "hsv_to_rgb" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( color, lab_to_rgb_matches_opencv )
{
  golden_image::expect_matches(
    "lab_to_rgb", io::lab_to_rgb( golden_image::input( "lab_to_rgb" ) ) );
}

// ----------------------------------------------------------------------------
/// A demosaic reads two pixels out, so the crop alone would show its own
/// edges. The mosaic is rebuilt at full size from the recorded RGB, the
/// demosaic runs on that, and the same window is compared.
TEST ( color, demosaic_matches_opencv )
{
  golden_image::expect_matches(
    "demosaic_bg",
    io::demosaic( golden_image::input( "demosaic_bg" ),
                  io::bayer_pattern::BG ) );
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give: what the conversions say about themselves.
TEST ( color, gray_of_a_flat_colour_is_that_colour )
{
  kv::image_of< uint8_t > flat( 3, 2, 3 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      flat( i, j, 0 ) = 90;
      flat( i, j, 1 ) = 90;
      flat( i, j, 2 ) = 90;
    }
  }

  auto const gray = io::rgb_to_gray( flat );

  for( size_t j = 0; j < gray.height(); ++j )
  {
    for( size_t i = 0; i < gray.width(); ++i )
    {
      EXPECT_EQ( 90, gray( i, j, 0 ) );
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( color, swap_rb_is_its_own_inverse )
{
  kv::image_of< uint8_t > image( 4, 3, 3 );
  uint8_t value = 0;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t p = 0; p < 3; ++p )
      {
        image( i, j, p ) = value++;
      }
    }
  }

  auto const twice = io::swap_rb( io::swap_rb( image ) );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t p = 0; p < 3; ++p )
      {
        EXPECT_EQ( image( i, j, p ), twice( i, j, p ) );
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// Alpha is not a colour, so `swap_rb` must leave it where it is.
TEST ( color, swap_rb_keeps_alpha )
{
  kv::image_of< uint8_t > image( 2, 1, 4 );
  image( 0, 0, 0 ) = 10; image( 0, 0, 1 ) = 20;
  image( 0, 0, 2 ) = 30; image( 0, 0, 3 ) = 40;
  image( 1, 0, 0 ) = 50; image( 1, 0, 1 ) = 60;
  image( 1, 0, 2 ) = 70; image( 1, 0, 3 ) = 80;

  auto const out = io::swap_rb( image );

  EXPECT_EQ( 30, out( 0, 0, 0 ) );
  EXPECT_EQ( 20, out( 0, 0, 1 ) );
  EXPECT_EQ( 10, out( 0, 0, 2 ) );
  EXPECT_EQ( 40, out( 0, 0, 3 ) );
  EXPECT_EQ( 80, out( 1, 0, 3 ) );
}

// ----------------------------------------------------------------------------
/// The float paths use the conventional ranges rather than the 8 bit ones.
TEST ( color, float_hsv_uses_real_degrees )
{
  kv::image_of< float > image( 1, 1, 3 );
  image( 0, 0, 0 ) = 0.0f;
  image( 0, 0, 1 ) = 1.0f;
  image( 0, 0, 2 ) = 0.0f;

  auto const hsv = io::rgb_to_hsv( image );

  EXPECT_NEAR( 120.0f, hsv( 0, 0, 0 ), 1e-4 );   // green
  EXPECT_NEAR( 1.0f, hsv( 0, 0, 1 ), 1e-4 );
  EXPECT_NEAR( 1.0f, hsv( 0, 0, 2 ), 1e-4 );

  auto const back = io::hsv_to_rgb( hsv );

  EXPECT_NEAR( 0.0f, back( 0, 0, 0 ), 1e-4 );
  EXPECT_NEAR( 1.0f, back( 0, 0, 1 ), 1e-4 );
  EXPECT_NEAR( 0.0f, back( 0, 0, 2 ), 1e-4 );
}

// ----------------------------------------------------------------------------
/// A mosaic of one flat colour demosaics to that colour, everywhere but the
/// border, whatever the pattern.
TEST ( color, flat_mosaic_demosaics_flat )
{
  for( auto const pattern : { io::bayer_pattern::BG, io::bayer_pattern::GB,
                              io::bayer_pattern::RG, io::bayer_pattern::GR } )
  {
    kv::image_of< uint8_t > mosaic( 8, 6, 1 );

    for( size_t j = 0; j < mosaic.height(); ++j )
    {
      for( size_t i = 0; i < mosaic.width(); ++i )
      {
        mosaic( i, j, 0 ) = 120;
      }
    }

    auto const rgb = io::demosaic( mosaic, pattern );

    ASSERT_EQ( 3u, rgb.depth() );

    for( size_t j = 1; j + 1 < rgb.height(); ++j )
    {
      for( size_t i = 1; i + 1 < rgb.width(); ++i )
      {
        for( size_t p = 0; p < 3; ++p )
        {
          EXPECT_EQ( 120, rgb( i, j, p ) )
            << "pattern " << int( pattern ) << " at (" << i << ", " << j
            << ", " << p << ")";
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// A demosaic returns the sample it was given at the site that holds it --
/// in the interior. The outermost ring is a different rule, below.
TEST ( color, demosaic_keeps_the_measured_sample )
{
  kv::image_of< uint8_t > mosaic( 6, 6, 1 );
  uint8_t value = 7;

  for( size_t j = 0; j < mosaic.height(); ++j )
  {
    for( size_t i = 0; i < mosaic.width(); ++i )
    {
      mosaic( i, j, 0 ) = static_cast< uint8_t >( value = ( value * 5 + 3 ) );
    }
  }

  auto const rgb = io::demosaic( mosaic, io::bayer_pattern::BG );

  // BG: blue at an even site, red at an odd one, green at the mixed pair
  EXPECT_EQ( mosaic( 2, 2, 0 ), rgb( 2, 2, 2 ) );
  EXPECT_EQ( mosaic( 4, 4, 0 ), rgb( 4, 4, 2 ) );
  EXPECT_EQ( mosaic( 1, 1, 0 ), rgb( 1, 1, 0 ) );
  EXPECT_EQ( mosaic( 3, 3, 0 ), rgb( 3, 3, 0 ) );
  EXPECT_EQ( mosaic( 2, 1, 0 ), rgb( 2, 1, 1 ) );
  EXPECT_EQ( mosaic( 1, 2, 0 ), rgb( 1, 2, 1 ) );
}

// ----------------------------------------------------------------------------
/// And the outermost ring does **not** keep it.
///
/// `cv::cvtColor` computes the interior and then copies the first and last
/// interior row and column outwards, rows before columns, so the sample the
/// sensor actually measured at (0, 0) is overwritten and the corner ends up
/// holding its diagonal neighbour. P7-T04b measured that against OpenCV --
/// this test used to assert the opposite and had been failing since, unrun,
/// because a discovered gtest carried no label. See finding 1.20.
TEST ( color, demosaic_replicates_the_outermost_ring )
{
  kv::image_of< uint8_t > mosaic( 6, 6, 1 );
  uint8_t value = 7;

  for( size_t j = 0; j < mosaic.height(); ++j )
  {
    for( size_t i = 0; i < mosaic.width(); ++i )
    {
      mosaic( i, j, 0 ) = static_cast< uint8_t >( value = ( value * 5 + 3 ) );
    }
  }

  auto const rgb = io::demosaic( mosaic, io::bayer_pattern::BG );

  auto const last = mosaic.width() - 1;

  for( size_t plane = 0; plane < 3; ++plane )
  {
    for( size_t i = 1; i + 1 < rgb.width(); ++i )
    {
      EXPECT_EQ( rgb( i, 0, plane ), rgb( i, 1, plane ) )
        << "top edge at column " << i << ", plane " << plane;
      EXPECT_EQ( rgb( i, last, plane ), rgb( i, last - 1, plane ) )
        << "bottom edge at column " << i << ", plane " << plane;
    }

    for( size_t j = 0; j < rgb.height(); ++j )
    {
      EXPECT_EQ( rgb( 0, j, plane ), rgb( 1, j, plane ) )
        << "left edge at row " << j << ", plane " << plane;
      EXPECT_EQ( rgb( last, j, plane ), rgb( last - 1, j, plane ) )
        << "right edge at row " << j << ", plane " << plane;
    }

    // Rows before columns, so a corner holds its diagonal neighbour
    EXPECT_EQ( rgb( 0, 0, plane ), rgb( 1, 1, plane ) );
    EXPECT_EQ( rgb( last, last, plane ), rgb( last - 1, last - 1, plane ) );
  }
}

// ----------------------------------------------------------------------------
TEST ( color, refuses_the_wrong_plane_count )
{
  kv::image_of< uint8_t > one( 2, 2, 1 );
  kv::image_of< uint8_t > three( 2, 2, 3 );

  EXPECT_THROW( io::rgb_to_gray( one ), std::invalid_argument );
  EXPECT_THROW( io::rgb_to_hsv( one ), std::invalid_argument );
  EXPECT_THROW( io::rgb_to_lab( one ), std::invalid_argument );
  EXPECT_THROW( io::swap_rb( one ), std::invalid_argument );
  EXPECT_THROW( io::demosaic( three, io::bayer_pattern::BG ),
                std::invalid_argument );
}
