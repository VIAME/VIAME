/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// The convolution kernels, against what OpenCV computed for the same pixels.
///
/// Same arrangement as `test_color.cxx`: `tests/golden/image_ops/opencv.json`
/// was recorded while OpenCV was still on the path, and each case carries its
/// own tolerance and the margin of border the comparison skips because the
/// recording is a window and a kernel reads outside it.
///
/// The kernel here runs on the window alone, so its edge is real -- which is
/// exactly why the margin exists. What is left after it is the interior,
/// where the two implementations see the same neighbourhood.

#include <image_ops/filter.h>

#include <viame/core_types/image.h>

#include "../golden_json.h"
#include "golden_image.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

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
TEST ( filter, gaussian_blur_matches_opencv )
{
  for( auto const& name : { "gaussian_blur_3", "gaussian_blur_5",
                            "gaussian_blur_5_rgb" } )
  {
    auto const size = ( std::string( name ) == "gaussian_blur_3" ) ? 3u : 5u;
    auto const input = golden_image::input( name );
    golden_image::expect_matches( name, io::gaussian_blur( input, size ) );
  }
}

// ----------------------------------------------------------------------------
/// A sigma given explicitly rather than derived from the size.
TEST ( filter, gaussian_blur_with_sigma_matches_opencv )
{
  auto const input = golden_image::input( "gaussian_blur_5_sigma_2" );
  golden_image::expect_matches( "gaussian_blur_5_sigma_2",
                                io::gaussian_blur( input, 5, 2.0 ) );
}

// ----------------------------------------------------------------------------
TEST ( filter, box_blur_matches_opencv )
{
  golden_image::expect_matches(
    "box_blur_3", io::box_blur( golden_image::input( "box_blur_3" ), 3 ) );
  golden_image::expect_matches(
    "box_blur_5", io::box_blur( golden_image::input( "box_blur_5" ), 5 ) );
}

// ----------------------------------------------------------------------------
/// The gradients, which are recorded signed and shifted by 32768 so that
/// they survive an unsigned JSON round trip.
TEST ( filter, sobel_matches_opencv )
{
  struct
  {
    char const* name;
    int dx;
    int dy;
    size_t size;
  } const cases[] = {
    { "sobel_dx_3", 1, 0, 3 },
    { "sobel_dy_3", 0, 1, 3 },
    { "sobel_dx_5", 1, 0, 5 },
    { "sobel_dxx_3", 2, 0, 3 },
  };

  for( auto const& one : cases )
  {
    auto const input = golden_image::input( one.name );
    auto const gradient =
      io::sobel< int16_t >( input, one.dx, one.dy, one.size );

    golden_image::expect_matches_signed( one.name, gradient );
  }
}

// ----------------------------------------------------------------------------
/// A kernel that is neither symmetric nor separable: a flipped kernel or a
/// transposed walk shows up here and nowhere else.
TEST ( filter, filter2d_matches_opencv )
{
  io::kernel sharpen;
  sharpen.width = 3;
  sharpen.height = 3;
  sharpen.weights = { 0.0, -1.0, 0.0,
                     -1.0,  5.0, -1.0,
                      0.0,  0.0,  1.0 };

  auto const input = golden_image::input( "filter2d_sharpen" );
  golden_image::expect_matches( "filter2d_sharpen",
                                io::filter_2d( input, sharpen ) );
}

// ----------------------------------------------------------------------------
/// The border rules, on a kernel wide enough that the edge dominates.
TEST ( filter, border_modes_match_opencv )
{
  struct
  {
    char const* name;
    io::border_mode mode;
  } const cases[] = {
    { "border_replicate", io::border_mode::REPLICATE },
    { "border_reflect", io::border_mode::REFLECT },
    { "border_reflect101", io::border_mode::REFLECT_101 },
    { "border_constant", io::border_mode::CONSTANT },
  };

  io::kernel line;
  line.width = 9;
  line.height = 1;
  line.weights.assign( 9, 1.0 / 9.0 );

  for( auto const& one : cases )
  {
    auto const input = golden_image::input( one.name );
    golden_image::expect_matches(
      one.name, io::filter_2d( input, line, one.mode ) );
  }
}

// ----------------------------------------------------------------------------
TEST ( filter, add_weighted_matches_opencv )
{
  auto const input = golden_image::input( "add_weighted" );
  auto const blurred = io::gaussian_blur( input, 5 );

  golden_image::expect_matches(
    "add_weighted", io::add_weighted( input, 1.5, blurred, -0.5 ) );
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give: what the kernels say about themselves.
TEST ( filter, a_blur_of_a_flat_image_is_that_image )
{
  kv::image_of< uint8_t > flat( 7, 5, 1 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      flat( i, j, 0 ) = 77;
    }
  }

  for( auto const size : { size_t{ 3 }, size_t{ 5 } } )
  {
    auto const blurred = io::gaussian_blur( flat, size );
    auto const boxed = io::box_blur( flat, size );

    for( size_t j = 0; j < flat.height(); ++j )
    {
      for( size_t i = 0; i < flat.width(); ++i )
      {
        EXPECT_EQ( 77, blurred( i, j, 0 ) ) << "gaussian " << size;
        EXPECT_EQ( 77, boxed( i, j, 0 ) ) << "box " << size;
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// A derivative of a flat image is zero, whatever the border rule.
TEST ( filter, a_gradient_of_a_flat_image_is_zero )
{
  kv::image_of< uint8_t > flat( 6, 4, 1 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      flat( i, j, 0 ) = 200;
    }
  }

  for( auto const mode : { io::border_mode::REPLICATE,
                           io::border_mode::REFLECT,
                           io::border_mode::REFLECT_101 } )
  {
    auto const dx = io::sobel< int16_t >( flat, 1, 0, 3, mode );
    auto const dy = io::sobel< int16_t >( flat, 0, 1, 3, mode );

    for( size_t j = 0; j < flat.height(); ++j )
    {
      for( size_t i = 0; i < flat.width(); ++i )
      {
        EXPECT_EQ( 0, dx( i, j, 0 ) ) << "mode " << int( mode );
        EXPECT_EQ( 0, dy( i, j, 0 ) ) << "mode " << int( mode );
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// A step edge has a gradient of one sign going up and the other coming down.
TEST ( filter, a_gradient_has_a_sign )
{
  kv::image_of< uint8_t > step( 6, 3, 1 );

  for( size_t j = 0; j < step.height(); ++j )
  {
    for( size_t i = 0; i < step.width(); ++i )
    {
      step( i, j, 0 ) = ( i < 3 ) ? 0 : 255;
    }
  }

  auto const dx = io::sobel< int16_t >( step, 1, 0, 3 );

  // Rising left to right, so the derivative across is positive at the step
  EXPECT_GT( dx( 2, 1, 0 ), 0 );
  EXPECT_GT( dx( 3, 1, 0 ), 0 );

  // and there is no vertical structure at all
  auto const dy = io::sobel< int16_t >( step, 0, 1, 3 );
  EXPECT_EQ( 0, dy( 1, 1, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( filter, gaussian_weights_sum_to_one )
{
  for( auto const size : { size_t{ 3 }, size_t{ 5 }, size_t{ 7 } } )
  {
    auto const line = io::gaussian_kernel_1d( size );

    double total = 0.0;
    for( auto const weight : line ) { total += weight; }

    EXPECT_NEAR( 1.0, total, 1e-12 ) << "size " << size;

    // and are symmetric about the centre
    for( size_t i = 0; i < size / 2; ++i )
    {
      EXPECT_NEAR( line[ i ], line[ size - 1 - i ], 1e-12 );
    }
  }
}

// ----------------------------------------------------------------------------
/// The Sobel rows OpenCV documents: (1, 2, 1) across and (-1, 0, 1) along.
TEST ( filter, sobel_rows_are_the_documented_ones )
{
  auto const smooth = io::sobel_kernel_1d( 0, 3 );
  ASSERT_EQ( 3u, smooth.size() );
  EXPECT_DOUBLE_EQ( 1.0, smooth[ 0 ] );
  EXPECT_DOUBLE_EQ( 2.0, smooth[ 1 ] );
  EXPECT_DOUBLE_EQ( 1.0, smooth[ 2 ] );

  auto const first = io::sobel_kernel_1d( 1, 3 );
  ASSERT_EQ( 3u, first.size() );
  EXPECT_DOUBLE_EQ( -1.0, first[ 0 ] );
  EXPECT_DOUBLE_EQ( 0.0, first[ 1 ] );
  EXPECT_DOUBLE_EQ( 1.0, first[ 2 ] );

  auto const second = io::sobel_kernel_1d( 2, 3 );
  ASSERT_EQ( 3u, second.size() );
  EXPECT_DOUBLE_EQ( 1.0, second[ 0 ] );
  EXPECT_DOUBLE_EQ( -2.0, second[ 1 ] );
  EXPECT_DOUBLE_EQ( 1.0, second[ 2 ] );
}

// ----------------------------------------------------------------------------
/// The border rules, spelled out on a row short enough to check by hand.
TEST ( filter, border_rules_are_what_they_say )
{
  kv::image_of< uint8_t > row( 5, 1, 1 );
  for( size_t i = 0; i < 5; ++i )
  {
    row( i, 0, 0 ) = static_cast< uint8_t >( 10 * ( i + 1 ) );  // 10..50
  }

  // A kernel that reads only two to the left, so the answer is the border
  io::kernel look_left;
  look_left.width = 5;
  look_left.height = 1;
  look_left.weights = { 1.0, 0.0, 0.0, 0.0, 0.0 };   // anchor is index 2

  auto const replicate =
    io::filter_2d( row, look_left, io::border_mode::REPLICATE );
  auto const reflect =
    io::filter_2d( row, look_left, io::border_mode::REFLECT );
  auto const reflect101 =
    io::filter_2d( row, look_left, io::border_mode::REFLECT_101 );
  auto const constant =
    io::filter_2d( row, look_left, io::border_mode::CONSTANT );

  // At i = 0 the kernel reads index -2
  EXPECT_EQ( 10, replicate( 0, 0, 0 ) );   // the edge, repeated
  EXPECT_EQ( 20, reflect( 0, 0, 0 ) );     // ...b a | a b c: index 1
  EXPECT_EQ( 30, reflect101( 0, 0, 0 ) );  // ...c b | a b c: index 2
  EXPECT_EQ( 0, constant( 0, 0, 0 ) );     // nothing

  // and in the middle every rule reads the same real pixel
  EXPECT_EQ( 10, replicate( 2, 0, 0 ) );
  EXPECT_EQ( 10, reflect( 2, 0, 0 ) );
  EXPECT_EQ( 10, reflect101( 2, 0, 0 ) );
  EXPECT_EQ( 10, constant( 2, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( filter, refuses_a_shapeless_kernel_or_a_mismatched_pair )
{
  kv::image_of< uint8_t > image( 3, 3, 1 );
  kv::image_of< uint8_t > other( 4, 3, 1 );

  io::kernel empty;
  EXPECT_THROW( io::filter_2d( image, empty ), std::invalid_argument );

  io::kernel lying;
  lying.width = 3;
  lying.height = 3;
  lying.weights.assign( 4, 1.0 );
  EXPECT_THROW( io::filter_2d( image, lying ), std::invalid_argument );

  EXPECT_THROW( io::gaussian_kernel_1d( 4 ), std::invalid_argument );
  EXPECT_THROW( io::gaussian_kernel_1d( 0 ), std::invalid_argument );
  EXPECT_THROW( io::sobel_kernel_1d( 1, 4 ), std::invalid_argument );
  EXPECT_THROW( io::sobel_kernel_1d( 3, 3 ), std::invalid_argument );

  EXPECT_THROW( io::add_weighted( image, 1.0, other, 1.0 ),
                std::invalid_argument );
}
