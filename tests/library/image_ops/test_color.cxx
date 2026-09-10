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

#include "../golden_json.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

using viame::testing::golden_json;

namespace {

/// The recording's directory, compiled in by CMake and overridable by the
/// environment so that a copy of the tree can be checked against another.
std::string
golden_path()
{
  char const* dir = std::getenv( "VIAME_GOLDEN_IMAGE_OPS_DIR" );

  if( !dir )
  {
    dir = VIAME_GOLDEN_IMAGE_OPS_DIR;
  }

  return std::string( dir ) + "/opencv.json";
}

/// One recorded image, as vital lays it out.
///
/// The recording is (row, column, plane) because that is numpy's order;
/// `image_of` is indexed (column, row, plane), so the walk transposes.
kv::image_of< uint8_t >
image_from( std::string const& object, std::string const& prefix )
{
  auto const width =
    static_cast< size_t >( golden_json::number( object, prefix + "_width" ) );
  auto const height =
    static_cast< size_t >( golden_json::number( object, prefix + "_height" ) );
  auto const planes =
    static_cast< size_t >( golden_json::number( object, prefix + "_planes" ) );

  auto const data = golden_json::numbers( object, prefix + "_data" );

  EXPECT_EQ( width * height * planes, data.size() ) << prefix;

  kv::image_of< uint8_t > out( width, height, planes );

  size_t at = 0;
  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < planes; ++p )
      {
        out( i, j, p ) = static_cast< uint8_t >( data[ at++ ] );
      }
    }
  }

  return out;
}

/// Every recorded case, by name.
std::vector< std::string > const&
cases()
{
  static std::vector< std::string > const all =
    golden_json( golden_path() ).section( "cases" );
  return all;
}

std::string
case_named( std::string const& name )
{
  for( auto const& object : cases() )
  {
    if( golden_json::text( object, "name" ) == name )
    {
      return object;
    }
  }

  ADD_FAILURE() << "no recorded case called '" << name << "'";
  return {};
}

/// Compare against the recording at the tolerance the recording states.
void
expect_matches( std::string const& name,
                kv::image_of< uint8_t > const& actual )
{
  auto const object = case_named( name );

  if( object.empty() )
  {
    return;
  }

  auto const expected = image_from( object, "expected" );
  auto const tolerance = golden_json::number( object, "tolerance" );

  // How many pixels of border the recording says to skip. OpenCV ran on the
  // whole fixture and the recording is a window of the result, so a kernel
  // that reads its neighbours sees this window's edge where OpenCV saw real
  // pixels. The recorder states the margin per case; only the neighbourhood
  // kernels have one.
  auto const margin =
    static_cast< size_t >( golden_json::number( object, "margin" ) );

  ASSERT_EQ( expected.width(), actual.width() ) << name;
  ASSERT_EQ( expected.height(), actual.height() ) << name;
  ASSERT_EQ( expected.depth(), actual.depth() ) << name;

  double worst = 0.0;
  double total = 0.0;

  size_t counted = 0;

  for( size_t j = margin; j + margin < expected.height(); ++j )
  {
    for( size_t i = margin; i + margin < expected.width(); ++i )
    {
      for( size_t p = 0; p < expected.depth(); ++p )
      {
        ++counted;

        auto const difference =
          std::abs( static_cast< double >( actual( i, j, p ) ) -
                    static_cast< double >( expected( i, j, p ) ) );

        if( difference > worst )
        {
          worst = difference;
        }

        total += difference;

        EXPECT_LE( difference, tolerance )
          << name << " at (" << i << ", " << j << ", " << p << "): "
          << int( actual( i, j, p ) ) << " against recorded "
          << int( expected( i, j, p ) );
      }
    }
  }

  // Printed rather than asserted on: a case that passes is more useful with
  // its margin visible, because a tolerance that is never approached is a
  // tolerance that could come down.
  ASSERT_GT( counted, 0u ) << name << ": the margin left nothing to compare";

  std::cout << "[          ] " << name << ": max " << worst << ", mean "
            << total / static_cast< double >( counted ) << " over " << counted
            << " values, tolerance " << tolerance
            << ( margin ? ", margin " + std::to_string( margin ) : "" )
            << std::endl;
}

/// The window the recording covers, taken out of a whole-image result.
///
/// The recorder runs each conversion on the whole fixture and records a crop,
/// so that a kernel reading its neighbours sees real ones. The test does the
/// same on its side.
constexpr size_t crop_left = 16;
constexpr size_t crop_top = 16;

kv::image_of< uint8_t >
cropped( kv::image_of< uint8_t > const& image, size_t width, size_t height )
{
  kv::image_of< uint8_t > out( width, height, image.depth() );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < image.depth(); ++p )
      {
        out( i, j, p ) = image( crop_left + i, crop_top + j, p );
      }
    }
  }

  return out;
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
/// The recorded input is the crop, so a per-pixel conversion needs no more.
TEST ( color, rgb_to_gray_matches_opencv )
{
  auto const object = case_named( "rgb_to_gray" );
  auto const input = image_from( object, "input" );
  expect_matches( "rgb_to_gray", io::rgb_to_gray( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, gray_to_rgb_matches_opencv )
{
  auto const object = case_named( "gray_to_rgb" );
  auto const input = image_from( object, "input" );
  expect_matches( "gray_to_rgb", io::gray_to_rgb( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, swap_rb_matches_opencv )
{
  auto const object = case_named( "swap_rb" );
  auto const input = image_from( object, "input" );
  expect_matches( "swap_rb", io::swap_rb( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, rgb_to_hsv_matches_opencv )
{
  auto const object = case_named( "rgb_to_hsv" );
  auto const input = image_from( object, "input" );
  expect_matches( "rgb_to_hsv", io::rgb_to_hsv( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, rgb_to_lab_matches_opencv )
{
  auto const object = case_named( "rgb_to_lab" );
  auto const input = image_from( object, "input" );
  expect_matches( "rgb_to_lab", io::rgb_to_lab( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, hsv_to_rgb_matches_opencv )
{
  auto const object = case_named( "hsv_to_rgb" );
  auto const input = image_from( object, "input" );
  expect_matches( "hsv_to_rgb", io::hsv_to_rgb( input ) );
}

// ----------------------------------------------------------------------------
TEST ( color, lab_to_rgb_matches_opencv )
{
  auto const object = case_named( "lab_to_rgb" );
  auto const input = image_from( object, "input" );
  expect_matches( "lab_to_rgb", io::lab_to_rgb( input ) );
}

// ----------------------------------------------------------------------------
/// A demosaic reads two pixels out, so the crop alone would show its own
/// edges. The mosaic is rebuilt at full size from the recorded RGB, the
/// demosaic runs on that, and the same window is compared.
TEST ( color, demosaic_matches_opencv )
{
  auto const object = case_named( "demosaic_bg" );
  auto const mosaic = image_from( object, "input" );

  expect_matches( "demosaic_bg",
                  io::demosaic( mosaic, io::bayer_pattern::BG ) );
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
/// A demosaic must return the sample it was given at the site that holds it.
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

  // BG: blue at (0,0), red at (1,1)
  EXPECT_EQ( mosaic( 0, 0, 0 ), rgb( 0, 0, 2 ) );
  EXPECT_EQ( mosaic( 2, 2, 0 ), rgb( 2, 2, 2 ) );
  EXPECT_EQ( mosaic( 1, 1, 0 ), rgb( 1, 1, 0 ) );
  EXPECT_EQ( mosaic( 3, 3, 0 ), rgb( 3, 3, 0 ) );
  // and green at the other two corners
  EXPECT_EQ( mosaic( 1, 0, 0 ), rgb( 1, 0, 1 ) );
  EXPECT_EQ( mosaic( 0, 1, 0 ), rgb( 0, 1, 1 ) );
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
