/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Histograms and the equalisations, against what OpenCV computed.
///
/// Recorded against the window rather than the whole fixture: an
/// equalisation is over the whole picture it is given, so a window of the
/// result is not the result on the window.

#include <image_ops/histogram.h>

#include <viame/core_types/image.h>

#include "golden_image.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
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
TEST ( histogram, normalize_matches_opencv )
{
  golden_image::expect_matches(
    "normalize_min_max",
    io::normalize_min_max( golden_image::input( "normalize_min_max" ),
                           0.0, 255.0 ) );
}

// ----------------------------------------------------------------------------
TEST ( histogram, equalize_matches_opencv )
{
  golden_image::expect_matches(
    "equalize_hist",
    io::equalize( golden_image::input( "equalize_hist" ) ) );
}

// ----------------------------------------------------------------------------
/// A large flat background is where the textbook mapping and OpenCV's differ
/// across the whole range rather than only at the ends.
TEST ( histogram, equalize_matches_opencv_on_a_flat_background )
{
  golden_image::expect_matches(
    "equalize_hist_flat_background",
    io::equalize(
      golden_image::input( "equalize_hist_flat_background" ) ) );
}

// ----------------------------------------------------------------------------
TEST ( histogram, clahe_matches_opencv )
{
  struct
  {
    char const* name;
    double clip;
    size_t tiles;
  } const cases[] = {
    { "clahe_clip_3_2x2", 3.0, 2 },
    { "clahe_clip_20_2x2", 20.0, 2 },
    { "clahe_clip_3_4x4", 3.0, 4 },
    { "clahe_clip_40_8x8", 40.0, 8 },
  };

  for( auto const& one : cases )
  {
    golden_image::expect_matches(
      one.name,
      io::clahe( golden_image::input( one.name ), one.clip, one.tiles,
                 one.tiles ) );
  }
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give.
TEST ( histogram, a_histogram_counts_every_pixel )
{
  kv::image_of< uint8_t > image( 5, 4, 1 );
  uint8_t value = 1;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( value = value * 3 + 7 );
    }
  }

  auto const counts = io::histogram_full( image );

  ASSERT_EQ( 256u, counts.size() );
  EXPECT_EQ( 20u, std::accumulate( counts.begin(), counts.end(), size_t{ 0 } ) );
}

// ----------------------------------------------------------------------------
/// The bins are half open, and the top value falls in the last one.
TEST ( histogram, the_top_value_lands_in_the_last_bin )
{
  kv::image_of< uint8_t > image( 3, 1, 1 );
  image( 0, 0, 0 ) = 0;
  image( 1, 0, 0 ) = 5;
  image( 2, 0, 0 ) = 10;

  auto const counts = io::histogram( image, 2, 0.0, 10.0 );

  ASSERT_EQ( 2u, counts.size() );
  EXPECT_EQ( 1u, counts[ 0 ] );   // 0
  EXPECT_EQ( 2u, counts[ 1 ] );   // 5 and 10
}

// ----------------------------------------------------------------------------
TEST ( histogram, min_max_finds_the_extremes_and_where_they_are )
{
  kv::image_of< uint8_t > image( 4, 3, 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = 100;
    }
  }

  image( 2, 1, 0 ) = 7;
  image( 0, 2, 0 ) = 250;

  io::extremum lowest;
  io::extremum highest;
  io::min_max( image, lowest, highest );

  EXPECT_DOUBLE_EQ( 7.0, lowest.value );
  EXPECT_EQ( 2u, lowest.i );
  EXPECT_EQ( 1u, lowest.j );

  EXPECT_DOUBLE_EQ( 250.0, highest.value );
  EXPECT_EQ( 0u, highest.i );
  EXPECT_EQ( 2u, highest.j );
}

// ----------------------------------------------------------------------------
/// A flat image has no range to stretch and comes back at the low end,
/// rather than dividing by zero.
TEST ( histogram, normalizing_a_flat_image_gives_the_low_end )
{
  kv::image_of< uint8_t > flat( 3, 2, 1 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      flat( i, j, 0 ) = 42;
    }
  }

  auto const out = io::normalize_min_max( flat, 10.0, 200.0 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      EXPECT_EQ( 10, out( i, j, 0 ) );
    }
  }
}

// ----------------------------------------------------------------------------
/// Normalising puts the extremes exactly at the ends asked for.
TEST ( histogram, normalizing_reaches_both_ends )
{
  kv::image_of< uint8_t > image( 4, 1, 1 );
  image( 0, 0, 0 ) = 40;
  image( 1, 0, 0 ) = 60;
  image( 2, 0, 0 ) = 80;
  image( 3, 0, 0 ) = 100;

  auto const out = io::normalize_min_max( image, 0.0, 255.0 );

  EXPECT_EQ( 0, out( 0, 0, 0 ) );
  EXPECT_EQ( 255, out( 3, 0, 0 ) );
  EXPECT_EQ( 85, out( 1, 0, 0 ) );
  EXPECT_EQ( 170, out( 2, 0, 0 ) );
}

// ----------------------------------------------------------------------------
/// Every value equal: nothing to flatten, and the image comes back as it is.
TEST ( histogram, equalizing_a_flat_image_changes_nothing )
{
  kv::image_of< uint8_t > flat( 4, 3, 1 );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      flat( i, j, 0 ) = 33;
    }
  }

  auto const out = io::equalize( flat );

  for( size_t j = 0; j < flat.height(); ++j )
  {
    for( size_t i = 0; i < flat.width(); ++i )
    {
      EXPECT_EQ( 33, out( i, j, 0 ) );
    }
  }

}

// ----------------------------------------------------------------------------
/// CLAHE on a flat image is where the clipping and the redistribution are
/// the whole answer, so it says exactly how the clipped remainder is given
/// back. OpenCV answers 85 at clip 3 with two tiles and 255 at clip 40 with
/// eight -- two different answers to the same question, which is why these
/// are recorded rather than reasoned about.
TEST ( histogram, clahe_on_a_flat_image_matches_opencv )
{
  golden_image::expect_matches(
    "clahe_flat_clip_3_2x2",
    io::clahe( golden_image::input( "clahe_flat_clip_3_2x2" ), 3.0, 2, 2 ) );

  golden_image::expect_matches(
    "clahe_flat_clip_40_8x8",
    io::clahe( golden_image::input( "clahe_flat_clip_40_8x8" ), 40.0, 8, 8 ) );
}

// ----------------------------------------------------------------------------
/// A size that does not divide by the tile grid, which is what a real frame
/// is. OpenCV pads when either dimension fails to divide and then pads both,
/// so a dimension that already divides gains a whole extra tile.
TEST ( histogram, clahe_on_an_odd_size_matches_opencv )
{
  golden_image::expect_matches(
    "clahe_flat_odd_size",
    io::clahe( golden_image::input( "clahe_flat_odd_size" ), 40.0, 4, 4 ) );

  golden_image::expect_matches(
    "clahe_odd_size",
    io::clahe( golden_image::input( "clahe_odd_size" ), 3.0, 4, 4 ) );
}

// ----------------------------------------------------------------------------
/// An equalisation is monotonic: a brighter pixel cannot come out darker.
TEST ( histogram, equalizing_preserves_order )
{
  kv::image_of< uint8_t > image( 16, 16, 1 );
  uint8_t value = 5;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      image( i, j, 0 ) = static_cast< uint8_t >( value = value * 7 + 11 );
    }
  }

  auto const out = io::equalize( image );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t jj = 0; jj < image.height(); ++jj )
      {
        for( size_t ii = 0; ii < image.width(); ++ii )
        {
          if( image( i, j, 0 ) < image( ii, jj, 0 ) )
          {
            ASSERT_LE( out( i, j, 0 ), out( ii, jj, 0 ) );
          }
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( histogram, refuses_what_it_cannot_do )
{
  kv::image_of< uint8_t > image( 3, 3, 1 );
  kv::image_of< uint8_t > colour( 3, 3, 3 );

  EXPECT_THROW( io::histogram( image, 0, 0.0, 255.0 ),
                std::invalid_argument );
  EXPECT_THROW( io::histogram( image, 8, 10.0, 10.0 ),
                std::invalid_argument );
  EXPECT_THROW( io::histogram( image, 8, 0.0, 255.0, 2 ),
                std::invalid_argument );

  EXPECT_THROW( io::clahe( colour ), std::invalid_argument );
  EXPECT_THROW( io::clahe( image, 3.0, 0, 2 ), std::invalid_argument );
}
