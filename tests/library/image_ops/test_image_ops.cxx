/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Unit tests for the image_ops kernels.
///
/// Every expected value here is computed by hand from the rule the kernel is
/// supposed to follow, not read back out of an implementation. The recordings
/// under tests/golden check the same kernels against what VXL actually did;
/// these check that what VXL did is what we think it did.

#include <image_ops/channels.h>
#include <image_ops/convert.h>
#include <image_ops/pixel.h>
#include <image_ops/statistics.h>
#include <image_ops/temporal.h>
#include <image_ops/threshold.h>

#include <vital/types/image.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace viame::image_ops;

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// An image whose pixels are the values given, in (i, j, plane) order.
template < typename T >
kv::image_of< T >
make_image( size_t width, size_t height, size_t depth,
            std::vector< T > const& values )
{
  kv::image_of< T > image( width, height, depth );
  size_t index = 0;

  for( size_t plane = 0; plane < depth; ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        image( i, j, plane ) = values[ index++ ];
      }
    }
  }

  return image;
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
TEST ( image_ops, cast_pixel_truncates_and_does_not_clamp )
{
  // A plain C++ conversion: toward zero, and no saturation
  EXPECT_EQ( 3, ( cast_pixel< uint8_t, double >( 3.9 ) ) );
  EXPECT_EQ( 3, ( cast_pixel< uint8_t, double >( 3.1 ) ) );
  EXPECT_EQ( 255, ( cast_pixel< uint8_t, uint16_t >( 255 ) ) );
  EXPECT_DOUBLE_EQ( 7.0, ( cast_pixel< double, uint8_t >( 7 ) ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, round_pixel_rounds_half_away_from_zero )
{
  EXPECT_EQ( 4, round_pixel< uint8_t >( 3.5 ) );
  EXPECT_EQ( 3, round_pixel< uint8_t >( 3.49 ) );
  EXPECT_EQ( -4, round_pixel< int16_t >( -3.5 ) );
  EXPECT_EQ( -3, round_pixel< int16_t >( -3.49 ) );

  // A real output type is left alone
  EXPECT_DOUBLE_EQ( 3.5, round_pixel< double >( 3.5 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, planes_to_grey_uses_bt709_weights )
{
  auto const image = make_image< uint8_t >(
    1, 1, 3, { 100, 200, 50 } );

  // 100*0.2125 + 200*0.7154 + 50*0.0721 = 21.25 + 143.08 + 3.605 = 167.935
  // rounded away from zero -> 168
  auto const grey = planes_to_grey( image );

  ASSERT_EQ( 1u, grey.depth() );
  EXPECT_EQ( 168, grey( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, mean_over_planes_truncates_in_the_pixel_type )
{
  // (10 + 11 + 11 + 11) / 4 = 10.75, truncated in uint8_t -> 10
  auto const image = make_image< uint8_t >( 1, 1, 4, { 10, 11, 11, 11 } );
  auto const mean = mean_over_planes( image );

  ASSERT_EQ( 1u, mean.depth() );
  EXPECT_EQ( 10, mean( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, mean_over_planes_wraps_around_in_the_pixel_type )
{
  // The accumulator is the pixel type, so 100 + 200 in uint8_t is 44, and
  // the mean of the two planes comes out 22 rather than 150. This is what
  // vil_math_mean_over_planes does and what the pipelines have always seen
  auto const image = make_image< uint8_t >( 1, 1, 2, { 100, 200 } );
  EXPECT_EQ( 22, mean_over_planes( image )( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, combine_channels_picks_luminance_only_for_three_planes )
{
  auto const rgb = make_image< uint8_t >( 1, 1, 3, { 100, 200, 50 } );
  EXPECT_EQ( 168, combine_channels( rgb )( 0, 0, 0 ) );

  // Two planes are averaged flat, in the pixel type: 100 + 200 wraps to 44,
  // halved to 22
  auto const pair = make_image< uint8_t >( 1, 1, 2, { 100, 200 } );
  EXPECT_EQ( 22, combine_channels( pair )( 0, 0, 0 ) );

  // Without the overflow the flat mean is what it looks like
  auto const small = make_image< uint8_t >( 1, 1, 2, { 10, 20 } );
  EXPECT_EQ( 15, combine_channels( small )( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, force_three_channels_repeats_plane_zero )
{
  auto const grey = make_image< uint8_t >( 2, 1, 1, { 7, 9 } );
  auto const three = force_three_channels( grey );

  ASSERT_EQ( 3u, three.depth() );
  for( size_t plane = 0; plane < 3; ++plane )
  {
    EXPECT_EQ( 7, three( 0, 0, plane ) );
    EXPECT_EQ( 9, three( 1, 0, plane ) );
  }

  // Two planes keep what they have and repeat plane 0 into the third
  auto const pair = make_image< uint8_t >( 1, 1, 2, { 4, 5 } );
  auto const grown = force_three_channels( pair );
  EXPECT_EQ( 4, grown( 0, 0, 0 ) );
  EXPECT_EQ( 5, grown( 0, 0, 1 ) );
  EXPECT_EQ( 4, grown( 0, 0, 2 ) );

  // Three planes are returned untouched
  auto const rgb = make_image< uint8_t >( 1, 1, 3, { 1, 2, 3 } );
  auto const same = force_three_channels( rgb );
  EXPECT_EQ( 1, same( 0, 0, 0 ) );
  EXPECT_EQ( 2, same( 0, 0, 1 ) );
  EXPECT_EQ( 3, same( 0, 0, 2 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, scale_rounds_and_saturates_at_the_top )
{
  auto const image = make_image< uint16_t >( 4, 1, 1, { 0, 100, 1000, 60000 } );

  // factor 0.5: 0, 50, 500 -> saturating at 255, and 60000 well past it
  auto const scaled = scale< uint8_t >( image, 0.5 );

  EXPECT_EQ( 0, scaled( 0, 0, 0 ) );
  EXPECT_EQ( 50, scaled( 1, 0, 0 ) );
  EXPECT_EQ( 255, scaled( 2, 0, 0 ) );
  EXPECT_EQ( 255, scaled( 3, 0, 0 ) );

  // Rounding is +0.5 then truncate: 3 * 0.5 = 1.5 -> 2
  auto const small = make_image< uint16_t >( 2, 1, 1, { 3, 5 } );
  auto const halved = scale< uint8_t >( small, 0.5 );
  EXPECT_EQ( 2, halved( 0, 0, 0 ) );
  EXPECT_EQ( 3, halved( 1, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, sample_and_sort_walks_every_pixel_of_a_single_plane )
{
  auto const image = make_image< uint8_t >( 2, 2, 1, { 4, 1, 3, 2 } );

  // Four pixels, four samples, stride one: every pixel, sorted
  auto const samples = sample_and_sort( image, 4 );
  ASSERT_EQ( 4u, samples.size() );
  EXPECT_EQ( 1, samples[ 0 ] );
  EXPECT_EQ( 2, samples[ 1 ] );
  EXPECT_EQ( 3, samples[ 2 ] );
  EXPECT_EQ( 4, samples[ 3 ] );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, sample_and_sort_strides_over_all_planes )
{
  // 2x2x2. The sample count is capped at one plane, four pixels, but the
  // stride is computed over the whole 8 pixel volume, so it is 8 / 4 = 2 and
  // the walk skips every other pixel. The position carries across planes.
  //
  // plane 0: values 10..13 at positions 0,1,2,3
  // plane 1: values 20..23
  //
  // plane 0 visits positions 0, 2, 4, 6 -> i,j = (0,0),(0,1),(0,0),(0,1)
  //   -> 10, 12, 10, 12
  // plane 1 continues at position 8, 10, 12, 14 -> (0,0),(0,1),(0,0),(0,1)
  //   -> 20, 22, 20, 22
  auto const image = make_image< uint8_t >(
    2, 2, 2, { 10, 11, 12, 13, 20, 21, 22, 23 } );

  auto const samples = sample_and_sort( image, 4 );

  ASSERT_EQ( 8u, samples.size() );
  EXPECT_EQ( 10, samples[ 0 ] );
  EXPECT_EQ( 10, samples[ 1 ] );
  EXPECT_EQ( 12, samples[ 2 ] );
  EXPECT_EQ( 12, samples[ 3 ] );
  EXPECT_EQ( 20, samples[ 4 ] );
  EXPECT_EQ( 20, samples[ 5 ] );
  EXPECT_EQ( 22, samples[ 6 ] );
  EXPECT_EQ( 22, samples[ 7 ] );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, sample_and_sort_can_drop_the_extremes )
{
  auto const image = make_image< uint8_t >(
    4, 1, 1, { 0, 7, 9, 255 } );

  auto const kept = sample_and_sort( image, 4, true );
  ASSERT_EQ( 2u, kept.size() );
  EXPECT_EQ( 7, kept[ 0 ] );
  EXPECT_EQ( 9, kept[ 1 ] );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, percentiles_index_by_rounded_fraction_of_the_last_index )
{
  auto const image = make_image< uint8_t >(
    5, 1, 1, { 10, 20, 30, 40, 50 } );

  // Five samples, last index 4. 0.0 -> 0, 0.5 -> 2, 1.0 -> 4.
  // 0.3 -> 4 * 0.3 + 0.5 = 1.7 -> 1
  auto const values = percentiles(
    image, std::vector< double >{ 0.0, 0.3, 0.5, 1.0 }, 5 );

  ASSERT_EQ( 4u, values.size() );
  EXPECT_EQ( 10, values[ 0 ] );
  EXPECT_EQ( 20, values[ 1 ] );
  EXPECT_EQ( 30, values[ 2 ] );
  EXPECT_EQ( 50, values[ 3 ] );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, percentile_stretch_maps_the_band_onto_the_output_range )
{
  auto const image = make_image< uint16_t >(
    5, 1, 1, { 100, 200, 300, 400, 500 } );

  // Extremes are only 0 and the type maximum, so nothing is dropped.
  // Bounds at 0.0 and 1.0 are 100 and 500, so the factor is
  // (255 + 0.5) / 400 = 0.63875, and each value maps to
  // trunc((v - 100) * 0.63875)
  auto const stretched = percentile_stretch< uint8_t >( image, 0.0, 1.0, 5 );

  EXPECT_EQ( 0, stretched( 0, 0, 0 ) );
  EXPECT_EQ( 63, stretched( 1, 0, 0 ) );   // 100 * 0.63875 = 63.875
  EXPECT_EQ( 127, stretched( 2, 0, 0 ) );  // 200 * 0.63875 = 127.75
  EXPECT_EQ( 191, stretched( 3, 0, 0 ) );  // 300 * 0.63875 = 191.625
  EXPECT_EQ( 255, stretched( 4, 0, 0 ) );  // 400 * 0.63875 = 255.5
}

// ----------------------------------------------------------------------------
TEST ( image_ops, percentile_stretch_clips_outside_the_band )
{
  auto const image = make_image< uint16_t >(
    5, 1, 1, { 100, 200, 300, 400, 500 } );

  // Bounds at 0.25 and 0.75 of the last index 4 are indices 1 and 3, so
  // 200 and 400. Below 200 -> 0, above 400 -> 255
  auto const stretched = percentile_stretch< uint8_t >( image, 0.25, 0.75, 5 );

  EXPECT_EQ( 0, stretched( 0, 0, 0 ) );
  EXPECT_EQ( 0, stretched( 1, 0, 0 ) );
  EXPECT_EQ( 255, stretched( 4, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, threshold_above_is_inclusive )
{
  // A pixel exactly at the threshold is kept, as vil_threshold_above does
  auto const image = make_image< uint8_t >( 3, 1, 1, { 9, 10, 11 } );
  auto const mask = threshold_above< uint8_t >( image, 10 );

  EXPECT_FALSE( mask( 0, 0, 0 ) );
  EXPECT_TRUE( mask( 1, 0, 0 ) );
  EXPECT_TRUE( mask( 2, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, threshold_percentile_cuts_at_the_sampled_value )
{
  auto const image = make_image< uint8_t >(
    5, 1, 1, { 10, 20, 30, 40, 50 } );

  // 0.5 of the last index 4 is index 2, value 30; at or above keeps 30, 40, 50
  auto const mask = threshold_percentile( image, 0.5, 5 );

  EXPECT_FALSE( mask( 0, 0, 0 ) );
  EXPECT_FALSE( mask( 1, 0, 0 ) );
  EXPECT_TRUE( mask( 2, 0, 0 ) );
  EXPECT_TRUE( mask( 3, 0, 0 ) );
  EXPECT_TRUE( mask( 4, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, cumulative_average_is_the_mean_of_every_frame )
{
  frame_averager< uint8_t > averager( average_mode::cumulative );

  auto const first = averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ) );
  EXPECT_EQ( 10, first( 0, 0, 0 ) );

  auto const second = averager.process( make_image< uint8_t >( 1, 1, 1, { 20 } ) );
  EXPECT_EQ( 15, second( 0, 0, 0 ) );

  // (10 + 20 + 30) / 3 = 20
  auto const third = averager.process( make_image< uint8_t >( 1, 1, 1, { 30 } ) );
  EXPECT_EQ( 20, third( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, cumulative_average_keeps_fractions_between_frames )
{
  frame_averager< uint8_t > averager( average_mode::cumulative );

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ) );
  // (10 + 11) / 2 = 10.5, truncated on the way out but kept inside
  auto const second = averager.process( make_image< uint8_t >( 1, 1, 1, { 11 } ) );
  EXPECT_EQ( 10, second( 0, 0, 0 ) );

  // (10 + 11 + 12) / 3 = 11 exactly, which only comes out right if the
  // accumulator kept the .5
  auto const third = averager.process( make_image< uint8_t >( 1, 1, 1, { 12 } ) );
  EXPECT_EQ( 11, third( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, cumulative_average_can_round_on_the_way_out )
{
  frame_averager< uint8_t > averager(
    average_mode::cumulative, 10, 0.3, /* round */ true );

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ) );
  // 10.5 rounds to 11 rather than truncating to 10
  auto const second = averager.process( make_image< uint8_t >( 1, 1, 1, { 11 } ) );
  EXPECT_EQ( 11, second( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, exponential_average_weights_the_new_frame )
{
  frame_averager< uint8_t > averager(
    average_mode::exponential, 10, 0.25 );

  averager.process( make_image< uint8_t >( 1, 1, 1, { 100 } ) );
  // 100 * 0.75 + 200 * 0.25 = 125
  auto const second =
    averager.process( make_image< uint8_t >( 1, 1, 1, { 200 } ) );
  EXPECT_EQ( 125, second( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, exponential_average_rejects_a_weight_outside_the_range )
{
  EXPECT_THROW( frame_averager< uint8_t >( average_mode::exponential, 10, 0.0 ),
                std::runtime_error );
  EXPECT_THROW( frame_averager< uint8_t >( average_mode::exponential, 10, 1.0 ),
                std::runtime_error );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, window_average_while_the_window_fills )
{
  frame_averager< uint8_t > averager( average_mode::window, 3 );

  EXPECT_EQ( 10, averager.process(
               make_image< uint8_t >( 1, 1, 1, { 10 } ) )( 0, 0, 0 ) );
  // (10 + 20) / 2 = 15
  EXPECT_EQ( 15, averager.process(
               make_image< uint8_t >( 1, 1, 1, { 20 } ) )( 0, 0, 0 ) );
  // (10 + 20 + 30) / 3 = 20
  EXPECT_EQ( 20, averager.process(
               make_image< uint8_t >( 1, 1, 1, { 30 } ) )( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, window_average_subtracts_the_newest_frame_once_full )
{
  // This pins the VXL behaviour described on frame_averager: with the window
  // full, the frame removed from the running sum is the most recently added
  // one, not the oldest, so the result is not the mean of the last N frames.
  frame_averager< uint8_t > averager( average_mode::window, 3 );

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ) );
  averager.process( make_image< uint8_t >( 1, 1, 1, { 20 } ) );
  averager.process( make_image< uint8_t >( 1, 1, 1, { 30 } ) );  // average 20

  // Full window. A true sliding mean of (20, 30, 40) would be 30.
  // What happens instead: 20 + (40 - 30) / 3 = 23.33, truncated to 23
  auto const fourth =
    averager.process( make_image< uint8_t >( 1, 1, 1, { 40 } ) );
  EXPECT_EQ( 23, fourth( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, variance_is_zero_on_the_first_frame )
{
  frame_averager< uint8_t > averager( average_mode::cumulative );
  kv::image_of< double > variance;

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ), variance );

  ASSERT_EQ( 1u, variance.width() );
  EXPECT_DOUBLE_EQ( 0.0, variance( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, variance_multiplies_the_distance_before_and_after )
{
  frame_averager< uint8_t > averager( average_mode::cumulative );
  kv::image_of< double > variance;

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ), variance );

  // Average before the update is 10, after it is 15.
  // |20 - 10| * |20 - 15| = 10 * 5 = 50
  averager.process( make_image< uint8_t >( 1, 1, 1, { 20 } ), variance );
  EXPECT_DOUBLE_EQ( 50.0, variance( 0, 0, 0 ) );
}

// ----------------------------------------------------------------------------
TEST ( image_ops, a_resolution_change_resets_the_average )
{
  frame_averager< uint8_t > averager( average_mode::cumulative );

  averager.process( make_image< uint8_t >( 1, 1, 1, { 10 } ) );

  // A different size starts over rather than mixing sizes
  auto const restarted =
    averager.process( make_image< uint8_t >( 2, 1, 1, { 40, 60 } ) );
  EXPECT_EQ( 40, restarted( 0, 0, 0 ) );
  EXPECT_EQ( 60, restarted( 1, 0, 0 ) );
}
