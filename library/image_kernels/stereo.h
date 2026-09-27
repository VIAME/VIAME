/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Semi-global block matching, as `cv::StereoSGBM` computes it
///
/// `MODE_SGBM` and `MODE_HH`. Identical to cv2 over 714240 pixels when this
/// landed -- 1440 configurations, both modes, heights 1 to 20, two disparity
/// counts, three block sizes, four uniqueness ratios, three `disp12MaxDiff`
/// values and two `preFilterCap` values.
///
/// Two details are worth reading before changing anything here, because both
/// look like mistakes and neither is. `design/lite-findings.md` 2.38 has the
/// measurements.
///
/// **The border value is `ftzero`, not zero.** OpenCV reaches its clip table
/// through a pointer that is already offset by `TAB_OFS`, so what its code
/// writes as `tab[0]` is the entry for a *difference of zero* -- `ftzero` --
/// and not the first entry of the table, which is the clamp at `-ftzero` and
/// comes out 0. Getting that wrong puts 0 where 15 belongs on the first and
/// last column of every row, of both channels, and moves 7.8% of the pixels.
///
/// **The recursion subtracts `minLr`, not `minLr + P2`,** even though
/// OpenCV's published source -- 5.x and the 5.0.0 tag, scalar and SIMD alike
/// -- subtracts the latter. Measured against the installed build, which is
/// what the goldens record, the former is right on every uniqueness ratio and
/// the latter only at zero. The two differ by a uniform `P2` in every `Lr`,
/// which is a fixed point of the recursion, so the aggregate cost comes out
/// `directions * P2` higher -- invisible to the winner and to the subpixel
/// fit, both of which are offset-invariant, and visible only to the
/// uniqueness ratio, which is a ratio rather than a difference.

#ifndef VIAME_IMAGE_KERNELS_STEREO_H
#define VIAME_IMAGE_KERNELS_STEREO_H

#include <image_kernels/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

/// `cv::StereoSGBM`'s three aggregations, which are three different algorithms
/// rather than three settings of one.
///
/// * `SGBM` -- one pass, five directions. `MODE_SGBM`, and the default.
/// * `HH` -- two passes, all eight directions, over the whole image at once.
///   `MODE_HH`; the most accurate and much the most memory.
/// * `SGBM_3WAY` -- three directions (left, the three above combined, right)
///   over **four horizontal stripes** with an overlap, which is what makes it
///   fast. `MODE_SGBM_3WAY`. The stripe count is fixed at four rather than
///   taken from the thread count, and OpenCV's comment says why: "to make the
///   results fully reproducible". The overlap is what lets each stripe's
///   top-down recursion settle before it reaches a row that gets written, and
///   it is why this is not the same answer as one stripe would give.
enum class sgbm_mode
{
  SGBM,
  HH,
  SGBM_3WAY,
};

/// What `cv::StereoSGBM_create` takes, with OpenCV's own defaults.
struct sgbm_params
{
  int min_disparity = 0;
  int num_disparities = 16;
  int block_size = 3;
  int p1 = 0;
  int p2 = 0;
  int disp12_max_diff = 0;
  int pre_filter_cap = 0;
  int uniqueness_ratio = 0;
  int speckle_window_size = 0;
  int speckle_range = 0;
  /// Which of OpenCV's three aggregations. See `sgbm_mode`.
  sgbm_mode mode = sgbm_mode::SGBM;
};

namespace detail {

constexpr int sgbm_disp_shift = 4;
constexpr int sgbm_disp_scale = 1 << sgbm_disp_shift;
constexpr int sgbm_max_cost = 32767;
constexpr int sgbm_tab_ofs = 256 * 4;

/// How many disparities OpenCV's three-way aggregation handles per vector.
///
/// Eight, because a `v_int16` is eight lanes at the **SSE2 baseline** and
/// `stereosgbm.cpp` is not one of OpenCV's dispatched files -- there is no
/// `sgbm.simd.hpp`, so it compiles once at whatever baseline the build chose
/// and never at AVX2's sixteen. That makes the lane-wise argmin in `three_way`
/// portable across x86-64 wheels rather than a property of this host, unlike
/// 2.56's `HSV2RGB` tail, which *is* in a dispatched file.
constexpr int sgbm_lanes = 8;

/// One row of the two matching "channels": a clipped Sobel and the row itself.
///
/// The rows above and below are clamped, which is what OpenCV's `n1`/`s1`
/// offsets of zero at the first and last row amount to.
/// The `2 * planes` rows a colour image contributes: a clipped Sobel per plane
/// and then the planes themselves, which is the order `diff_scale` keys off.
inline void
sgbm_prepare( viame::image_of< uint8_t > const& image, size_t y,
              std::vector< int > const& tab, int ftzero,
              std::vector< std::vector< int > >& rows )
{
  auto const width = image.width();
  auto const height = image.height();
  auto const planes = image.depth();
  auto const above = ( y > 0 ) ? y - 1 : y;
  auto const below = ( y + 1 < height ) ? y + 1 : y;

  rows.assign( planes * 2, std::vector< int >( width, ftzero ) );

  auto const at = [ & ]( size_t row, size_t x, size_t plane )
  {
    return static_cast< int >( image( x, row, plane ) );
  };

  for( size_t plane = 0; plane < planes; ++plane )
  {
    auto& sobel = rows[ plane ];
    auto& raw = rows[ planes + plane ];

    for( size_t x = 1; x + 1 < width; ++x )
    {
      auto const value =
        2 * ( at( y, x + 1, plane ) - at( y, x - 1, plane ) ) +
        ( at( above, x + 1, plane ) - at( above, x - 1, plane ) ) +
        ( at( below, x + 1, plane ) - at( below, x - 1, plane ) );

      sobel[ x ] = tab[ static_cast< size_t >( value + sgbm_tab_ofs ) ];
      raw[ x ] = at( y, x, plane );
    }
  }
}

/// The half-sample extremes Birchfield-Tomasi compares against.
inline void
sgbm_extremes( std::vector< int > const& row, std::vector< int >& lowest,
               std::vector< int >& highest )
{
  auto const width = row.size();

  lowest.resize( width );
  highest.resize( width );

  for( size_t x = 0; x < width; ++x )
  {
    auto const value = row[ x ];
    auto const left = ( x > 0 ) ? ( value + row[ x - 1 ] ) / 2 : value;
    auto const right = ( x + 1 < width ) ? ( value + row[ x + 1 ] ) / 2 : value;

    lowest[ x ] = std::min( std::min( left, right ), value );
    highest[ x ] = std::max( std::max( left, right ), value );
  }
}

/// `cv::filterSpeckles`: blank any region smaller than \p most whose
/// neighbours are within \p tolerance of each other.
///
/// A flood per unlabelled pixel, where a neighbour joins the region when it
/// is within `tolerance` of **the pixel being expanded** rather than of the
/// seed -- so a region is a chain of small steps and can span a range wider
/// than the tolerance. The set it reaches does not depend on the traversal
/// order, which is why this uses a plain stack where OpenCV walks a
/// wavefront.
inline void
filter_speckles( viame::image_of< int16_t >& image, int blank, int most,
                 int tolerance )
{
  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );

  if( width == 0 || height == 0 )
  {
    return;
  }

  std::vector< int > label( static_cast< size_t >( width ) * height, 0 );
  std::vector< bool > speck( 1, false );
  std::vector< std::pair< int, int > > pending;

  auto const index = [ & ]( int x, int y )
  {
    return static_cast< size_t >( y ) * width + x;
  };
  auto const value = [ & ]( int x, int y )
  {
    return static_cast< int >(
      image( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) );
  };

  auto current = 0;

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      if( value( x, y ) == blank )
      {
        continue;
      }

      if( label[ index( x, y ) ] != 0 )
      {
        if( speck[ static_cast< size_t >( label[ index( x, y ) ] ) ] )
        {
          image( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
            static_cast< int16_t >( blank );
        }

        continue;
      }

      ++current;
      speck.push_back( false );

      label[ index( x, y ) ] = current;
      pending.clear();
      pending.emplace_back( x, y );

      auto size = 0;

      while( !pending.empty() )
      {
        auto const at = pending.back();
        pending.pop_back();
        ++size;

        auto const here = value( at.first, at.second );

        int const around[ 4 ][ 2 ] = { { at.first, at.second + 1 },
                                       { at.first, at.second - 1 },
                                       { at.first + 1, at.second },
                                       { at.first - 1, at.second } };

        for( auto const& next : around )
        {
          auto const nx = next[ 0 ];
          auto const ny = next[ 1 ];

          if( nx < 0 || ny < 0 || nx >= width || ny >= height ||
              label[ index( nx, ny ) ] != 0 )
          {
            continue;
          }

          auto const there = value( nx, ny );

          if( there == blank || std::abs( here - there ) > tolerance )
          {
            continue;
          }

          label[ index( nx, ny ) ] = current;
          pending.emplace_back( nx, ny );
        }
      }

      if( size <= most )
      {
        speck[ static_cast< size_t >( current ) ] = true;
        image( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
          static_cast< int16_t >( blank );
      }
    }
  }
}


/// `cv::saturate_cast< CostType >`, CostType being a signed short.
inline int
saturate_cost( int value )
{
  return std::min( std::max( value, -32768 ), 32767 );
}

/// Saturating 16-bit add and subtract, which is what OpenCV's `v_add` and
/// `v_sub` do on a `v_int16`.
///
/// This matters and is easy to miss. `stereosgbm.cpp` carries a scalar
/// reference beside every vectorised loop, and the two are **not** the same
/// arithmetic: the scalar one promotes to `int`, so `cost + P1` on a cost near
/// 32767 keeps growing, where the vector one saturates there. OpenCV's
/// universal intrinsics define `+` on an 8 or 16 bit lane as the *saturating*
/// instruction and spell the wrapping one `v_add_wrap`. Reading the scalar path
/// as the specification therefore gets the answer wrong on exactly the pixels
/// whose aggregated cost reaches the top of the type -- about half a percent of
/// a 512 by 512 frame, which is few enough to look like a tie break and is not.
inline int
add_cost( int a, int b )
{
  return saturate_cost( a + b );
}

inline int
sub_cost( int a, int b )
{
  return saturate_cost( a - b );
}

/// `MODE_SGBM_3WAY`'s aggregation: three directions over four stripes.
///
/// The cost volume is the same one the other two modes use; what differs is the
/// aggregation. Two passes over each row -- one left to right that also carries
/// the top-down recursion, and one right to left that sums all three and picks
/// the winner -- run over **four horizontal stripes** whose top few rows are
/// scratch. The stripe count is fixed, not taken from the thread count, so the
/// result does not depend on the machine.
///
/// The overlap is what makes this an approximation rather than a
/// reorganisation: each stripe's top-down recursion starts from nothing at its
/// first row and needs a few rows to settle, so the rows it will actually write
/// begin `overlap` rows later. A single stripe would give a different answer,
/// and a different overlap would too.
inline void
three_way( std::vector< int > const& across, int width, int height, int first,
           int last, int count, int min_d, int half, int p1, int p2,
           int uniqueness, int max_diff, int16_t invalid,
           viame::image_of< int16_t >& out )
{
  auto const span = last - first;
  auto const lanes = static_cast< size_t >( count );

  // Four stripes, and the overlap OpenCV computes for them.
  constexpr int stripes = 4;
  auto const stripe_size = static_cast< int >(
    std::ceil( static_cast< double >( height ) / stripes ) );
  auto const overlap = ( half + 1 ) +
    static_cast< int >( std::ceil( 0.1 * stripe_size ) );

  // `horizontal` and `vertical` are indexed from one rather than zero, so that
  // the recursion can read the previous column without a branch; the zeroth
  // slot stays zero for the whole run, which is what makes the first column's
  // `min` come out as P2 and cancel the `+ p2` on the cost below.
  std::vector< int > horizontal( static_cast< size_t >( span + 2 ) * lanes, 0 );
  std::vector< int > vertical( static_cast< size_t >( span + 2 ) * lanes, 0 );
  std::vector< int > vertical_min( static_cast< size_t >( span + 2 ), 0 );
  std::vector< int > right( lanes, 0 );
  std::vector< int > disp2( static_cast< size_t >( width ) );
  std::vector< int > disp2cost( static_cast< size_t >( width ) );

  // Each stripe writes into its own buffer and the whole map is assembled from
  // the four afterwards, exactly as OpenCV does it -- and the indexing is
  // OpenCV's too, quirk included. A stripe writes row `y` at buffer row
  // `(stripe == 0 ? overlap : 0) + (y - begin)`, while the assembly reads
  // buffer row `overlap + i % stripe_size`. Those agree only while
  // `stripe * stripe_size >= overlap`, which holds for any frame taller than a
  // few dozen rows and fails for a very short one -- and there the assembly
  // reads buffer rows nothing wrote, which come back as "no disparity".
  // Reproduced rather than corrected: it is what a caller gets from cv2.
  auto const buffer_rows = stripe_size + overlap;
  std::vector< std::vector< int16_t > > buffers(
    static_cast< size_t >( stripes ),
    std::vector< int16_t >(
      static_cast< size_t >( buffer_rows ) * width, invalid ) );

  for( int stripe = 0; stripe < stripes; ++stripe )
  {
    auto const begin = std::max(
      std::min( stripe * stripe_size - overlap, height ), 0 );
    auto const end = std::min( ( stripe + 1 ) * stripe_size, height );

    if( begin >= end )
    {
      continue;
    }

    // Everything the recursion carries starts fresh for each stripe.
    std::fill( horizontal.begin(), horizontal.end(), 0 );
    std::fill( vertical.begin(), vertical.end(), 0 );
    std::fill( vertical_min.begin(), vertical_min.end(), 0 );

    // Where this stripe's rows land in its own buffer.
    auto const offset = ( stripe == 0 ) ? overlap : 0;
    auto& buffer = buffers[ static_cast< size_t >( stripe ) ];

    // The vertical box, clamped at **this stripe's** first row rather than at
    // the image's. That only differs for the scratch rows -- the overlap is
    // always more than half a block, so a row that gets written has its whole
    // box inside the stripe -- but the scratch rows are what the top-down
    // recursion starts from, so the difference reaches the written rows anyway.
    std::vector< int > cost( static_cast< size_t >( end - begin ) * span *
                             count, 0 );

    for( int y = begin; y < end; ++y )
    {
      for( int k = -half; k <= half; ++k )
      {
        auto const at = std::min( std::max( y + k, begin ), height - 1 );
        auto* into = &cost[ static_cast< size_t >( y - begin ) * span * count ];
        auto const* from =
          &across[ static_cast< size_t >( at ) * span * count ];

        for( int i = 0; i < span * count; ++i ) { into[ i ] += from[ i ]; }
      }
    }

    for( int y = begin; y < end; ++y )
    {
      auto const row_in_buffer = offset + ( y - begin );
      auto const writing = row_in_buffer < buffer_rows;
      auto* disp_row = writing
        ? &buffer[ static_cast< size_t >( row_in_buffer ) * width ] : nullptr;

      for( int x = 0; x < width; ++x )
      {
        disp2[ static_cast< size_t >( x ) ] = invalid;
        disp2cost[ static_cast< size_t >( x ) ] = sgbm_max_cost;
      }

      auto const* row_cost =
        &cost[ static_cast< size_t >( y - begin ) * span * count ];

      // Left to right, and top to bottom in the same sweep.
      auto left_min = 0;

      for( int x = 0; x < span; ++x )
      {
        auto const at = static_cast< size_t >( x + 1 ) * lanes;
        auto const previous = static_cast< size_t >( x ) * lanes;
        auto const* costs = row_cost + static_cast< size_t >( x ) * count;

        auto& top_min = vertical_min[ static_cast< size_t >( x + 1 ) ];
        auto const left_ceiling = saturate_cost( left_min + p2 );
        auto const top_ceiling = saturate_cost( top_min + p2 );

        auto left_new = sgbm_max_cost;
        auto top_new = sgbm_max_cost;
        auto left_before = sgbm_max_cost;
        auto top_before = sgbm_max_cost;

        for( int d = 0; d < count; ++d )
        {
          auto const last_one = d == count - 1;
          // The cost carries a `+ p2` that the `- left_ceiling` below takes
          // straight back off at the first column, where the previous row of
          // the buffer is all zero. OpenCV gets the same by initialising its
          // cost volume line to P2 rather than to nothing.
          auto const value = saturate_cost( costs[ d ] + p2 );

          // The order is the vector body's, one saturating step at a time:
          // the two neighbours are reduced first and P1 added to the winner,
          // rather than added to each.
          auto const left_up = last_one ? sgbm_max_cost
            : horizontal[ previous + static_cast< size_t >( d ) + 1 ];
          auto const left_here = horizontal[ previous +
                                             static_cast< size_t >( d ) ];
          auto const left_found = add_cost(
            value,
            sub_cost(
              std::min( add_cost( std::min( left_before, left_up ), p1 ),
                        std::min( left_here, left_ceiling ) ),
              left_ceiling ) );

          left_before = left_here;
          horizontal[ at + static_cast< size_t >( d ) ] = left_found;
          left_new = std::min( left_new, left_found );

          auto const top_here = vertical[ at + static_cast< size_t >( d ) ];
          auto const top_up = last_one ? sgbm_max_cost
            : vertical[ at + static_cast< size_t >( d ) + 1 ];
          auto const top_found = add_cost(
            value,
            sub_cost(
              std::min( add_cost( std::min( top_before, top_up ), p1 ),
                        std::min( top_here, top_ceiling ) ),
              top_ceiling ) );

          top_before = top_here;
          vertical[ at + static_cast< size_t >( d ) ] = top_found;
          top_new = std::min( top_new, top_found );
        }

        left_min = left_new;
        top_min = top_new;
      }

      // Right to left, summing the three and taking the winner.
      std::fill( right.begin(), right.end(), 0 );

      auto right_min = 0;

      for( int x = span - 1; x >= 0; --x )
      {
        auto const at = static_cast< size_t >( x + 1 ) * lanes;
        auto const* costs = row_cost + static_cast< size_t >( x ) * count;

        auto const right_ceiling = saturate_cost( right_min + p2 );
        auto right_new = sgbm_max_cost;
        auto right_before = sgbm_max_cost;

        for( int d = 0; d < count; ++d )
        {
          auto const value = saturate_cost( costs[ d ] + p2 );
          auto const right_here = right[ static_cast< size_t >( d ) ];
          auto const right_up = ( d == count - 1 ) ? sgbm_max_cost
            : right[ static_cast< size_t >( d ) + 1 ];
          auto const found = add_cost(
            value,
            sub_cost(
              std::min( add_cost( std::min( right_before, right_up ), p1 ),
                        std::min( right_here, right_ceiling ) ),
              right_ceiling ) );

          right_before = right_here;
          right[ static_cast< size_t >( d ) ] = found;
          right_new = std::min( right_new, found );

          // Two saturating adds, left to right, as the vector body writes it.
          auto const total = add_cost(
            add_cost( found, horizontal[ at + static_cast< size_t >( d ) ] ),
            vertical[ at + static_cast< size_t >( d ) ] );

          horizontal[ at + static_cast< size_t >( d ) ] = total;

        }

        right_min = right_new;

        // The winning disparity, and **not** simply the lowest-cost one.
        //
        // OpenCV's argmin is lane-wise. `min_sum_cost_reg` carries a running
        // minimum per lane across blocks of `sgbm_lanes` disparities, and
        // `min_sum_pos_reg` carries the base of the **last** block in which
        // that lane matched its minimum; `min_pos` then reduces to the lowest
        // `base + lane` among the lanes holding the overall minimum. So a tie
        // between two disparities in the same lane goes to the **later** one
        // and a tie across lanes to the lower lane -- which is why neither
        // "lowest wins" nor "highest wins" reproduces it, and why fitting one
        // of those looked right on one image and wrong on another.
        auto lowest = sgbm_max_cost;
        auto best = 0;

        {
          auto const* totals_row = &horizontal[ at ];
          auto const aligned =
            ( ( count + sgbm_lanes - 1 ) / sgbm_lanes ) * sgbm_lanes;
          // Where the vector body stops. It runs while `i < Da - lanes`, and
          // the extra block that finishes the job exists only when `Da` is
          // `count` exactly -- otherwise a scalar tail takes the remainder.
          auto const vector_end =
            ( count == aligned ) ? count : aligned - sgbm_lanes;

          int lane_min[ sgbm_lanes ];
          int lane_base[ sgbm_lanes ];
          bool lane_used[ sgbm_lanes ];

          for( int lane = 0; lane < sgbm_lanes; ++lane )
          {
            lane_min[ lane ] = sgbm_max_cost;
            lane_base[ lane ] = 0;
            lane_used[ lane ] = false;
          }

          for( int d = 0; d < vector_end; ++d )
          {
            auto const lane = d % sgbm_lanes;
            auto const base = d - lane;
            auto const total = totals_row[ d ];

            if( !lane_used[ lane ] || total < lane_min[ lane ] )
            {
              lane_min[ lane ] = total;
              lane_base[ lane ] = base;
              lane_used[ lane ] = true;
            }
            else if( total == lane_min[ lane ] )
            {
              lane_base[ lane ] = base;
            }
          }

          for( int lane = 0; lane < sgbm_lanes; ++lane )
          {
            if( lane_used[ lane ] && lane_min[ lane ] < lowest )
            {
              lowest = lane_min[ lane ];
            }
          }

          auto found = sgbm_max_cost;

          for( int lane = 0; lane < sgbm_lanes; ++lane )
          {
            if( lane_used[ lane ] && lane_min[ lane ] == lowest )
            {
              found = std::min( found, lane_base[ lane ] + lane );
            }
          }

          best = ( found == sgbm_max_cost ) ? 0 : found;

          // The scalar tail, strictly less, which only runs when `count` is
          // not a whole number of lanes.
          for( int d = vector_end; d < count; ++d )
          {
            if( totals_row[ d ] < lowest )
            {
              lowest = totals_row[ d ];
              best = d;
            }
          }
        }

        if( !writing )
        {
          continue;
        }

        auto const* totals_at = &horizontal[ at ];

        if( uniqueness > 0 )
        {
          // A **truncating threshold**, not the ratio inequality the other two
          // modes use. OpenCV computes `thresh = 100 * min / (100 - ratio)` in
          // int and then compares `total < (short)( thresh + 1 )`, so the test
          // is `total <= floor( ... )` where the ratio form is a strict `<` --
          // they part company on every pixel where `total * (100 - ratio)`
          // equals `100 * min` exactly. **And the cast wraps**: past a minimum
          // of about 29490 the threshold exceeds a signed short, comes back
          // negative, and no disparity qualifies, so the pixel survives a test
          // the ratio form would have failed it on. Both together were the last
          // 251 pixels of a 512 by 512 frame.
          auto const limit = static_cast< int >( static_cast< int16_t >(
            ( 100 * lowest ) / ( 100 - uniqueness ) + 1 ) );

          auto d = 0;

          for( ; d < count; ++d )
          {
            if( totals_at[ d ] < limit && std::abs( d - best ) > 1 )
            {
              break;
            }
          }

          if( d < count )
          {
            continue;
          }
        }

        auto d = best;
        auto const mirrored = x + first - d - min_d;

        if( mirrored >= 0 && mirrored < width &&
            disp2cost[ static_cast< size_t >( mirrored ) ] > lowest )
        {
          disp2cost[ static_cast< size_t >( mirrored ) ] = lowest;
          disp2[ static_cast< size_t >( mirrored ) ] = d + min_d;
        }

        int scaled;

        if( d > 0 && d < count - 1 )
        {
          auto const denom = std::max(
            totals_at[ d - 1 ] + totals_at[ d + 1 ] - 2 * totals_at[ d ], 1 );
          auto const numerator =
            ( totals_at[ d - 1 ] - totals_at[ d + 1 ] ) * sgbm_disp_scale +
            denom;

          scaled = d * sgbm_disp_scale + numerator / ( denom * 2 );
        }
        else
        {
          scaled = d * sgbm_disp_scale;
        }

        disp_row[ x + first ] =
          static_cast< int16_t >( scaled + min_d * sgbm_disp_scale );
      }

      if( !writing )
      {
        continue;
      }

      for( int x = first; x < last; ++x )
      {
        auto const value = static_cast< int >( disp_row[ x ] );

        if( value == invalid )
        {
          continue;
        }

        auto const down = value >> sgbm_disp_shift;
        auto const up = ( value + sgbm_disp_scale - 1 ) >> sgbm_disp_shift;
        auto const a = x - down;
        auto const b = x - up;

        if( a >= 0 && a < width &&
            disp2[ static_cast< size_t >( a ) ] >= min_d &&
            std::abs( disp2[ static_cast< size_t >( a ) ] - down ) > max_diff &&
            b >= 0 && b < width &&
            disp2[ static_cast< size_t >( b ) ] >= min_d &&
            std::abs( disp2[ static_cast< size_t >( b ) ] - up ) > max_diff )
        {
          disp_row[ x ] = invalid;
        }
      }
    }
  }

  // The assembly, by OpenCV's rule rather than by where each row was written.
  for( int y = 0; y < height; ++y )
  {
    auto const stripe = std::min( y / stripe_size, stripes - 1 );
    auto const row = overlap + ( y % stripe_size );
    auto const& buffer = buffers[ static_cast< size_t >( stripe ) ];

    for( int x = 0; x < width; ++x )
    {
      out( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        ( row < buffer_rows )
        ? buffer[ static_cast< size_t >( row ) * width + x ] : invalid;
    }
  }
}

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::StereoSGBM::compute`: a disparity map in 1/16 of a pixel.
///
/// The result is signed 16 bit, as OpenCV's is, with
/// `(min_disparity - 1) * 16` meaning "no disparity here".
inline viame::image_of< int16_t >
stereo_sgbm( viame::image_of< uint8_t > const& left,
             viame::image_of< uint8_t > const& right,
             sgbm_params const& params )
{
  using namespace detail;

  if( left.depth() != right.depth() ||
      ( left.depth() != 1 && left.depth() != 3 ) )
  {
    throw std::invalid_argument(
      "stereo_sgbm takes one or three plane images, matching" );
  }

  if( left.width() != right.width() || left.height() != right.height() )
  {
    throw std::invalid_argument( "stereo_sgbm: the two images differ in size" );
  }

  auto const width = static_cast< int >( left.width() );
  auto const height = static_cast< int >( left.height() );

  auto const min_d = params.min_disparity;
  auto const max_d = min_d + params.num_disparities;
  auto const count = max_d - min_d;

  auto const half = ( params.block_size > 0 ) ? params.block_size / 2 : 1;
  auto const ftzero = std::max( params.pre_filter_cap, 15 ) | 1;
  auto const p1 = ( params.p1 > 0 ) ? params.p1 : 2;
  auto const p2 = std::max( ( params.p2 > 0 ) ? params.p2 : 5, p1 + 1 );
  auto const uniqueness = ( params.uniqueness_ratio >= 0 )
                          ? params.uniqueness_ratio : 10;
  auto const max_diff = ( params.disp12_max_diff > 0 )
                        ? params.disp12_max_diff : 1;

  auto const invalid = static_cast< int16_t >(
    ( min_d - 1 ) * sgbm_disp_scale );

  viame::image_of< int16_t > out(
    static_cast< size_t >( width ), static_cast< size_t >( height ), 1 );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      out( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) = invalid;
    }
  }

  auto const first = std::max( max_d, 0 );
  auto const last = width + std::min( min_d, 0 );
  auto const span = last - first;

  if( width == 0 || height == 0 || count <= 0 || span <= 0 )
  {
    return out;
  }

  // The clip table, indexed through `+ sgbm_tab_ofs` -- see the file comment
  // on why its zero entry is not its first entry.
  std::vector< int > tab( 256 + sgbm_tab_ofs * 2 );

  for( size_t i = 0; i < tab.size(); ++i )
  {
    auto const value = static_cast< int >( i ) - sgbm_tab_ofs;

    tab[ i ] = std::min( std::max( value, -ftzero ), ftzero ) + ftzero;
  }

  // The per-pixel cost of every row, then the SAD box over it.
  std::vector< int > raw_cost(
    static_cast< size_t >( height ) * span * count, 0 );

  {
    auto const planes = static_cast< int >( left.depth() );
    std::vector< std::vector< int > > l_rows, r_rows;
    std::vector< int > l_low, l_high, r_low, r_high;

    for( int y = 0; y < height; ++y )
    {
      sgbm_prepare( left, static_cast< size_t >( y ), tab, ftzero, l_rows );
      sgbm_prepare( right, static_cast< size_t >( y ), tab, ftzero, r_rows );

      for( int channel = 0; channel < planes * 2; ++channel )
      {
        auto const& u_row = l_rows[ static_cast< size_t >( channel ) ];
        auto const& v_row = r_rows[ static_cast< size_t >( channel ) ];
        // The Sobels first at full weight, then the raw planes at a quarter.
        auto const shift = ( channel < planes ) ? 0 : 2;

        sgbm_extremes( u_row, l_low, l_high );
        sgbm_extremes( v_row, r_low, r_high );

        for( int x = first; x < last; ++x )
        {
          auto const u = u_row[ static_cast< size_t >( x ) ];
          auto const u0 = l_low[ static_cast< size_t >( x ) ];
          auto const u1 = l_high[ static_cast< size_t >( x ) ];

          auto* into = &raw_cost[ ( static_cast< size_t >( y ) * span +
                                    ( x - first ) ) * count ];

          for( int d = 0; d < count; ++d )
          {
            auto const at = x - ( min_d + d );
            auto const v = v_row[ static_cast< size_t >( at ) ];
            auto const v0 = r_low[ static_cast< size_t >( at ) ];
            auto const v1 = r_high[ static_cast< size_t >( at ) ];

            auto const c0 = std::max( 0, std::max( u - v1, v0 - u ) );
            auto const c1 = std::max( 0, std::max( v - u1, u0 - v ) );

            into[ d ] += std::min( c0, c1 ) >> shift;
          }
        }
      }
    }
  }

  std::vector< int > cost( raw_cost.size(), 0 );
  std::vector< int > across( raw_cost.size(), 0 );

  {
    for( int y = 0; y < height; ++y )
    {
      for( int x = 0; x < span; ++x )
      {
        auto* into = &across[ ( static_cast< size_t >( y ) * span + x ) * count ];

        for( int k = -half; k <= half; ++k )
        {
          auto const at = std::min( std::max( x + k, 0 ), span - 1 );
          auto const* from = &raw_cost[ ( static_cast< size_t >( y ) * span +
                                          at ) * count ];

          for( int d = 0; d < count; ++d ) { into[ d ] += from[ d ]; }
        }
      }
    }

    for( int y = 0; y < height; ++y )
    {
      for( int k = -half; k <= half; ++k )
      {
        auto const at = std::min( std::max( y + k, 0 ), height - 1 );

        for( int x = 0; x < span; ++x )
        {
          auto* into = &cost[ ( static_cast< size_t >( y ) * span + x ) * count ];
          auto const* from = &across[ ( static_cast< size_t >( at ) * span +
                                        x ) * count ];

          for( int d = 0; d < count; ++d ) { into[ d ] += from[ d ]; }
        }
      }
    }
  }

  if( params.mode == sgbm_mode::SGBM_3WAY )
  {
    // `across` rather than `cost`: the vertical box is rebuilt per stripe,
    // because its top clamps at the stripe's first row and not at the image's.
    three_way( across, width, height, first, last, count, min_d, half, p1, p2,
               uniqueness, max_diff, invalid, out );
  }
  else
  {

  auto const passes = ( params.mode == sgbm_mode::HH ) ? 2 : 1;
  std::vector< int > totals( cost.size(), 0 );

  // Lr is (index, direction, disparity), with the disparity padded by one at
  // each end so that d-1 and d+1 are always addressable, and the index padded
  // so that -1 and `span` are.
  auto const lanes = static_cast< size_t >( count + 2 );
  auto const stride = lanes * 4;
  std::vector< std::vector< int > > lr( 2 );
  std::vector< std::vector< int > > min_lr( 2 );

  std::vector< int > disp2( static_cast< size_t >( width ) );
  std::vector< int > disp2cost( static_cast< size_t >( width ) );

  for( int pass = 1; pass <= passes; ++pass )
  {
    auto const step = ( pass == 1 ) ? 1 : -1;
    auto const y_from = ( pass == 1 ) ? 0 : height - 1;
    auto const x_from = ( pass == 1 ) ? 0 : span - 1;

    for( int which = 0; which < 2; ++which )
    {
      lr[ which ].assign( stride * ( span + 2 ), 0 );
      min_lr[ which ].assign( 4u * ( span + 2 ), 0 );
    }

    int id = 0;

    for( int n = 0; n < height; ++n )
    {
      auto const y = y_from + n * step;

      if( pass == 1 )
      {
        for( int x = 0; x < span; ++x )
        {
          auto* into = &totals[ ( static_cast< size_t >( y ) * span + x ) * count ];

          for( int d = 0; d < count; ++d ) { into[ d ] = 0; }
        }
      }

      auto const slot = [ & ]( int which, int index, int direction ) -> int*
      {
        return &lr[ which ][ stride * static_cast< size_t >( index + 1 ) +
                             lanes * static_cast< size_t >( direction ) ];
      };
      auto const lowest = [ & ]( int which, int index, int direction ) -> int&
      {
        return min_lr[ which ][ 4u * static_cast< size_t >( index + 1 ) +
                                static_cast< size_t >( direction ) ];
      };

      for( int m = 0; m < span; ++m )
      {
        auto const x = x_from + m * step;

        int* previous[ 4 ] = { slot( id, x - step, 0 ),
                               slot( 1 - id, x - 1, 1 ),
                               slot( 1 - id, x, 2 ),
                               slot( 1 - id, x + 1, 3 ) };
        int const floor_of[ 4 ] = { lowest( id, x - step, 0 ),
                                    lowest( 1 - id, x - 1, 1 ),
                                    lowest( 1 - id, x, 2 ),
                                    lowest( 1 - id, x + 1, 3 ) };

        for( int k = 0; k < 4; ++k )
        {
          previous[ k ][ 0 ] = sgbm_max_cost;
          previous[ k ][ count + 1 ] = sgbm_max_cost;
        }

        auto const* here = &cost[ ( static_cast< size_t >( y ) * span + x ) *
                                  count ];
        auto* into = &totals[ ( static_cast< size_t >( y ) * span + x ) * count ];

        int best[ 4 ] = { sgbm_max_cost, sgbm_max_cost, sgbm_max_cost,
                          sgbm_max_cost };

        for( int d = 0; d < count; ++d )
        {
          auto sum = into[ d ];

          for( int k = 0; k < 4; ++k )
          {
            auto const* row = previous[ k ];
            auto const ceiling = floor_of[ k ] + p2;
            auto const value =
              here[ d ] +
              std::min( std::min( row[ d + 1 ], row[ d ] + p1 ),
                        std::min( row[ d + 2 ] + p1, ceiling ) ) -
              floor_of[ k ];

            slot( id, x, k )[ d + 1 ] = value;
            best[ k ] = std::min( best[ k ], value );
            sum += value;
          }

          into[ d ] = std::max( -32768, std::min( 32767, sum ) );
        }

        for( int k = 0; k < 4; ++k ) { lowest( id, x, k ) = best[ k ]; }
      }

      if( pass == passes )
      {
        for( int x = 0; x < width; ++x )
        {
          disp2[ static_cast< size_t >( x ) ] = invalid;
          disp2cost[ static_cast< size_t >( x ) ] = sgbm_max_cost;
        }

        for( int x = span - 1; x >= 0; --x )
        {
          auto* totals_at = &totals[ ( static_cast< size_t >( y ) * span + x ) *
                                     count ];
          auto lowest_total = sgbm_max_cost;
          auto winner = -1;

          if( passes == 1 )
          {
            auto* row = slot( id, x + 1, 0 );
            row[ 0 ] = row[ count + 1 ] = sgbm_max_cost;

            auto const floor_here = lowest( id, x + 1, 0 );
            auto const ceiling = floor_here + p2;
            auto const* here = &cost[ ( static_cast< size_t >( y ) * span + x ) *
                                      count ];
            auto best_fifth = sgbm_max_cost;

            for( int d = 0; d < count; ++d )
            {
              auto const value =
                here[ d ] +
                std::min( std::min( row[ d + 1 ], row[ d ] + p1 ),
                          std::min( row[ d + 2 ] + p1, ceiling ) ) -
                floor_here;

              slot( id, x, 0 )[ d + 1 ] = value;
              best_fifth = std::min( best_fifth, value );

              auto const total = std::max(
                -32768, std::min( 32767, totals_at[ d ] + value ) );

              totals_at[ d ] = total;

              if( total < lowest_total )
              {
                lowest_total = total;
                winner = d;
              }
            }

            lowest( id, x, 0 ) = best_fifth;
          }
          else
          {
            for( int d = 0; d < count; ++d )
            {
              if( totals_at[ d ] < lowest_total )
              {
                lowest_total = totals_at[ d ];
                winner = d;
              }
            }
          }

          auto crowded = false;

          for( int d = 0; d < count; ++d )
          {
            if( totals_at[ d ] * ( 100 - uniqueness ) < lowest_total * 100 &&
                std::abs( winner - d ) > 1 )
            {
              crowded = true;
              break;
            }
          }

          if( crowded )
          {
            continue;
          }

          auto d = winner;
          auto const mirrored = x + first - d - min_d;

          if( mirrored >= 0 && mirrored < width &&
              disp2cost[ static_cast< size_t >( mirrored ) ] > lowest_total )
          {
            disp2cost[ static_cast< size_t >( mirrored ) ] = lowest_total;
            disp2[ static_cast< size_t >( mirrored ) ] = d + min_d;
          }

          int scaled;

          if( d > 0 && d < count - 1 )
          {
            // The quadratic fit, with C's truncating division rather than a
            // floor: the numerator goes negative and the two differ there.
            auto const denom =
              std::max( totals_at[ d - 1 ] + totals_at[ d + 1 ] -
                        2 * totals_at[ d ], 1 );
            auto const numerator =
              ( totals_at[ d - 1 ] - totals_at[ d + 1 ] ) * sgbm_disp_scale +
              denom;

            scaled = d * sgbm_disp_scale + numerator / ( denom * 2 );
          }
          else
          {
            scaled = d * sgbm_disp_scale;
          }

          out( static_cast< size_t >( x + first ),
               static_cast< size_t >( y ), 0 ) =
            static_cast< int16_t >( scaled + min_d * sgbm_disp_scale );
        }

        for( int x = first; x < last; ++x )
        {
          auto const value = static_cast< int >(
            out( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) );

          if( value == invalid )
          {
            continue;
          }

          auto const down = value >> sgbm_disp_shift;
          auto const up = ( value + sgbm_disp_scale - 1 ) >> sgbm_disp_shift;
          auto const a = x - down;
          auto const b = x - up;

          if( a >= 0 && a < width &&
              disp2[ static_cast< size_t >( a ) ] >= min_d &&
              std::abs( disp2[ static_cast< size_t >( a ) ] - down ) > max_diff &&
              b >= 0 && b < width &&
              disp2[ static_cast< size_t >( b ) ] >= min_d &&
              std::abs( disp2[ static_cast< size_t >( b ) ] - up ) > max_diff )
          {
            out( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
              invalid;
          }
        }
      }

      id = 1 - id;
    }
  }

  } // the SGBM and HH aggregations

  // The 3 by 3 median SGBM finishes with, borders replicating.
  viame::image_of< int16_t > smoothed(
    static_cast< size_t >( width ), static_cast< size_t >( height ), 1 );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      int window[ 9 ];
      auto at = 0;

      for( int dy = -1; dy <= 1; ++dy )
      {
        for( int dx = -1; dx <= 1; ++dx )
        {
          auto const sy = std::min( std::max( y + dy, 0 ), height - 1 );
          auto const sx = std::min( std::max( x + dx, 0 ), width - 1 );

          window[ at++ ] = out( static_cast< size_t >( sx ),
                                static_cast< size_t >( sy ), 0 );
        }
      }

      std::nth_element( window, window + 4, window + 9 );

      smoothed( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        static_cast< int16_t >( window[ 4 ] );
    }
  }

  // And the speckle filter, after the median rather than before it.
  if( params.speckle_window_size > 0 )
  {
    filter_speckles( smoothed, invalid, params.speckle_window_size,
                     params.speckle_range * sgbm_disp_scale );
  }

  return smoothed;
}

} // namespace image_kernels
} // namespace viame

#endif
