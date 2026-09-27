/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The fast global smoother, and the disparity WLS filter over it
///
/// `cv::ximgproc::fastGlobalSmootherFilter` and
/// `cv::ximgproc::DisparityWLSFilter`. Three shipped stereo pipelines set
/// `use_wls_filter true`, which is the only reason the disparity computer still
/// reached for cv2 once all three SGBM aggregations were exact.
///
/// The smoother is Min et al.'s "Fast Global Image Smoothing Based on Weighted
/// Least Squares": rather than solve the two-dimensional system, it alternates
/// a horizontal and a vertical **tridiagonal** solve and attenuates lambda
/// between iterations. Each row's solve is independent of every other row's, so
/// unlike SGBM's three-way mode there is no stripe approximation here and the
/// answer does not depend on how the work was divided.
///
/// The weights come from a **lookup table of 3 * 256 * 256 entries** indexed by
/// the squared colour difference between neighbouring guide pixels, holding
/// `-exp( -sqrt( i ) / sigma )`. The negative sign is folded into the table
/// rather than into the recursion, which is why the tridiagonal coefficients
/// below read as `1 - a - b` where the paper writes a sum.

#ifndef VIAME_IMAGE_KERNELS_SMOOTHER_H
#define VIAME_IMAGE_KERNELS_SMOOTHER_H

#include <image_kernels/color.h>
#include <image_kernels/filter.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

namespace detail {

/// `-exp( -sqrt( i ) / sigma )` for every squared colour difference.
///
/// Three channels of a byte difference squared is at most 3 * 255 * 255, and
/// the table covers 3 * 256 * 256 so that the index never has to be checked.
inline std::vector< float >
smoother_weights( double sigma )
{
  constexpr int levels = 3 * 256 * 256;
  std::vector< float > out( static_cast< size_t >( levels ) );

  for( int i = 0; i < levels; ++i )
  {
    out[ static_cast< size_t >( i ) ] = -std::exp(
      -std::sqrt( static_cast< float >( i ) ) /
      static_cast< float >( sigma ) );
  }

  return out;
}

/// The squared difference between two guide pixels, summed over the planes.
inline int
smoother_difference( viame::image_of< uint8_t > const& guide,
                     size_t ax, size_t ay, size_t bx, size_t by )
{
  auto total = 0;

  for( size_t plane = 0; plane < guide.depth(); ++plane )
  {
    auto const gap = static_cast< int >( guide( ax, ay, plane ) ) -
                     static_cast< int >( guide( bx, by, plane ) );

    total += gap * gap;
  }

  return total;
}

/// One tridiagonal solve along a line, which is the whole of the smoother.
///
/// `weight[ k ]` is the edge between sample `k` and `k + 1`, and the last is
/// zero because there is no edge past the end. The forward sweep is Thomas's
/// algorithm with the elimination written out; the backward sweep substitutes.
inline void
smoother_line( std::vector< float >& values, std::vector< float > const& weight,
               std::vector< float >& scratch, float lambda, int count )
{
  if( count <= 0 ) { return; }

  auto previous = lambda * weight[ 0 ];

  scratch[ 0 ] = previous / ( 1.0f - previous );
  values[ 0 ] = values[ 0 ] / ( 1.0f - previous );

  for( int j = 1; j < count; ++j )
  {
    auto const current = lambda * weight[ static_cast< size_t >( j ) ];

    // Written exactly as OpenCV's scalar path writes it, because two
    // plausible-looking changes each make it worse and the measurements are
    // worth recording: summing the two coefficients first, which is what its
    // *vector* path does, takes a 24-configuration sweep from 12 differing to
    // 22; fusing the two multiply-subtracts takes it to 22 and the worst
    // relative gap from 1.1e-05 to 1.1e-04. This is a recursive solve, so a
    // single last-bit difference at one end of a row compounds along it --
    // which is also why half the sweep is exact and the rest is not, rather
    // than all of it being a little off.
    auto const denominator = ( 1.0f - previous - current ) -
      scratch[ static_cast< size_t >( j ) - 1 ] * previous;

    scratch[ static_cast< size_t >( j ) ] = current / denominator;
    values[ static_cast< size_t >( j ) ] =
      ( values[ static_cast< size_t >( j ) ] -
        values[ static_cast< size_t >( j ) - 1 ] * previous ) / denominator;
    previous = current;
  }

  for( int j = count - 2; j >= 0; --j )
  {
    values[ static_cast< size_t >( j ) ] -=
      scratch[ static_cast< size_t >( j ) ] *
      values[ static_cast< size_t >( j ) + 1 ];
  }
}

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::ximgproc::fastGlobalSmootherFilter`, edge-aware and separable.
///
/// \p guide is one or three planes of bytes and gives the edges; \p image is
/// what gets smoothed, in float. `lambda` is the strength, `sigma` the colour
/// scale of an edge, and each iteration multiplies lambda by \p attenuation --
/// OpenCV's defaults are three iterations at a quarter.
inline viame::image_of< float >
smooth_globally( viame::image_of< uint8_t > const& guide,
                 viame::image_of< float > const& image, double lambda,
                 double sigma, double attenuation = 0.25, int iterations = 3 )
{
  if( guide.width() != image.width() || guide.height() != image.height() )
  {
    throw std::invalid_argument(
      "smooth_globally: the guide and the image differ in size" );
  }

  if( guide.depth() != 1 && guide.depth() != 3 )
  {
    throw std::invalid_argument(
      "smooth_globally: the guide is one or three planes" );
  }

  if( iterations < 1 )
  {
    throw std::invalid_argument(
      "smooth_globally: at least one iteration" );
  }

  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );
  auto const planes = image.depth();

  viame::image_of< float > out( image.width(), image.height(), planes );

  if( width == 0 || height == 0 || planes == 0 )
  {
    return out;
  }

  auto const table = detail::smoother_weights( sigma );

  // The edge weights, once: horizontal between (x, y) and (x + 1, y), vertical
  // between (x, y) and (x, y + 1), and zero along the far edge of each.
  std::vector< float > horizontal(
    static_cast< size_t >( width ) * height, 0.0f );
  std::vector< float > vertical(
    static_cast< size_t >( width ) * height, 0.0f );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x + 1 < width; ++x )
    {
      horizontal[ static_cast< size_t >( y ) * width + x ] =
        table[ static_cast< size_t >( detail::smoother_difference(
          guide, static_cast< size_t >( x ), static_cast< size_t >( y ),
          static_cast< size_t >( x + 1 ), static_cast< size_t >( y ) ) ) ];
    }
  }

  for( int y = 0; y + 1 < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      vertical[ static_cast< size_t >( y ) * width + x ] =
        table[ static_cast< size_t >( detail::smoother_difference(
          guide, static_cast< size_t >( x ), static_cast< size_t >( y ),
          static_cast< size_t >( x ), static_cast< size_t >( y + 1 ) ) ) ];
    }
  }

  auto const longest = static_cast< size_t >( std::max( width, height ) );
  std::vector< float > line( longest );
  std::vector< float > weights( longest );
  std::vector< float > scratch( longest );
  std::vector< float > current(
    static_cast< size_t >( width ) * height );

  for( size_t plane = 0; plane < planes; ++plane )
  {
    for( int y = 0; y < height; ++y )
    {
      for( int x = 0; x < width; ++x )
      {
        current[ static_cast< size_t >( y ) * width + x ] =
          image( static_cast< size_t >( x ), static_cast< size_t >( y ),
                 plane );
      }
    }

    auto strength = static_cast< float >( lambda );

    for( int pass = 0; pass < iterations; ++pass )
    {
      for( int y = 0; y < height; ++y )
      {
        auto const row = static_cast< size_t >( y ) * width;

        for( int x = 0; x < width; ++x )
        {
          line[ static_cast< size_t >( x ) ] =
            current[ row + static_cast< size_t >( x ) ];
          weights[ static_cast< size_t >( x ) ] =
            horizontal[ row + static_cast< size_t >( x ) ];
        }

        detail::smoother_line( line, weights, scratch, strength, width );

        for( int x = 0; x < width; ++x )
        {
          current[ row + static_cast< size_t >( x ) ] =
            line[ static_cast< size_t >( x ) ];
        }
      }

      for( int x = 0; x < width; ++x )
      {
        for( int y = 0; y < height; ++y )
        {
          line[ static_cast< size_t >( y ) ] =
            current[ static_cast< size_t >( y ) * width + x ];
          weights[ static_cast< size_t >( y ) ] =
            vertical[ static_cast< size_t >( y ) * width + x ];
        }

        detail::smoother_line( line, weights, scratch, strength, height );

        for( int y = 0; y < height; ++y )
        {
          current[ static_cast< size_t >( y ) * width + x ] =
            line[ static_cast< size_t >( y ) ];
        }
      }

      strength *= static_cast< float >( attenuation );
    }

    for( int y = 0; y < height; ++y )
    {
      for( int x = 0; x < width; ++x )
      {
        out( static_cast< size_t >( x ), static_cast< size_t >( y ), plane ) =
          current[ static_cast< size_t >( y ) * width + x ];
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// What `cv::ximgproc::DisparityWLSFilter` needs beside the two disparity maps.
struct wls_params
{
  double lambda = 8000.0;
  double sigma = 1.0;
  /// `min_disparity + num_disparities` and `-min_disparity`, both floored at
  /// zero, which is the band of columns a disparity can be trusted in.
  int left_offset = 0;
  int right_offset = 0;
  int min_disparity = 0;
  /// `ceil( 0.5 * block_size )` for SGBM.
  int discontinuity_radius = 5;
  int lrc_threshold = 24;
  float roll_off = 0.001f;
};

// ----------------------------------------------------------------------------
/// `cv::ximgproc::DisparityWLSFilter::filter` with the confidence map on.
///
/// The two disparity maps are the left matcher's and the right matcher's, both
/// in sixteenths as `stereo_sgbm` gives them. The result is a float disparity,
/// still in sixteenths, smoothed towards the edges of \p guide and weighted by
/// a confidence that falls off where the two views disagree or where the
/// disparity is locally rough.
///
/// The confidence is two things multiplied into one map. A **depth
/// discontinuity** term, `1 - roll_off * variance` over a box around each
/// pixel, which distrusts a disparity sitting on a jump; and a **left-right
/// consistency** term, which zeroes any pixel whose partner in the right map
/// does not agree within `lrc_threshold`. Then the filter smooths
/// `confidence * disparity` and `confidence` separately and divides one by the
/// other, so a low-confidence pixel takes its value from its neighbours.
inline viame::image_of< float >
filter_disparity_wls( viame::image_of< uint8_t > const& guide,
                      viame::image_of< int16_t > const& left_disparity,
                      viame::image_of< int16_t > const& right_disparity,
                      wls_params const& params )
{
  auto const width = static_cast< int >( left_disparity.width() );
  auto const height = static_cast< int >( left_disparity.height() );

  if( static_cast< int >( right_disparity.width() ) != width ||
      static_cast< int >( right_disparity.height() ) != height ||
      static_cast< int >( guide.width() ) != width ||
      static_cast< int >( guide.height() ) != height )
  {
    throw std::invalid_argument(
      "filter_disparity_wls: the guide and the two disparity maps differ in "
      "size" );
  }

  viame::image_of< float > out( static_cast< size_t >( width ),
                                static_cast< size_t >( height ), 1 );

  auto const blank =
    static_cast< float >( 16 * ( params.min_disparity - 1 ) );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      out( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) = blank;
    }
  }

  if( width == 0 || height == 0 )
  {
    return out;
  }

  // The band the left matcher can have found anything in, and its mirror for
  // the right one.
  auto const left_x = std::min( std::max( params.left_offset, 0 ), width );
  auto const left_span = std::max(
    0, width - params.left_offset - params.right_offset );
  auto const right_x = width - ( left_x + left_span );

  auto const left_at = [ & ]( int x, int y )
  {
    return static_cast< float >(
      left_disparity( static_cast< size_t >( x ), static_cast< size_t >( y ),
                      0 ) );
  };
  auto const right_at = [ & ]( int x, int y )
  {
    return static_cast< float >(
      right_disparity( static_cast< size_t >( x ), static_cast< size_t >( y ),
                       0 ) );
  };

  // The depth discontinuity maps, from the variance of the disparity over a
  // box -- `E[d^2] - E[d]^2`, each term a plain box mean.
  auto const radius = params.discontinuity_radius;
  auto const side = 2 * radius + 1;

  auto const discontinuity =
    [ & ]( int origin, bool from_left ) -> std::vector< float >
    {
      std::vector< float > out_map(
        static_cast< size_t >( left_span ) * height, 0.0f );

      for( int y = 0; y < height; ++y )
      {
        for( int x = 0; x < left_span; ++x )
        {
          auto mean = 0.0;
          auto square = 0.0;

          for( int dy = -radius; dy <= radius; ++dy )
          {
            for( int dx = -radius; dx <= radius; ++dx )
            {
              // `cv::boxFilter`'s default border, reflect-101, over the band.
              auto const sy = static_cast< int >( detail::border_index(
                y + dy, height, border_mode::REFLECT_101 ) );
              auto const sx = static_cast< int >( detail::border_index(
                x + dx, left_span, border_mode::REFLECT_101 ) );
              auto const value = from_left
                ? left_at( origin + sx, sy ) : right_at( origin + sx, sy );

              mean += value;
              square += static_cast< double >( value ) * value;
            }
          }

          auto const count = static_cast< double >( side ) * side;
          auto const average = static_cast< float >( mean / count );
          auto const variance =
            static_cast< float >( square / count ) - average * average;

          out_map[ static_cast< size_t >( y ) * left_span + x ] =
            std::max( 1.0f - params.roll_off * variance, 0.0f );
        }
      }

      return out_map;
    };

  auto const left_disc = discontinuity( left_x, true );
  auto const right_disc = discontinuity( right_x, false );

  // The confidence **starts as** the left discontinuity map, and the left-right
  // check then overwrites only the pixels whose partner in the right map is
  // inside the band. That is not a detail: a pixel whose partner falls off the
  // edge -- the rightmost `num_disparities` columns, mostly -- keeps its
  // discontinuity value and is trusted, where zeroing it instead throws away
  // 1634 pixels of a 512 by 512 frame. OpenCV gets this by assigning the map
  // into `confidence_map` before the pass and leaving the `if` without an
  // `else`; it is easy to read as a bug and it is load bearing.
  viame::image_of< float > confidence( static_cast< size_t >( width ),
                                       static_cast< size_t >( height ), 1 );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      auto const inside = x >= left_x && x < left_x + left_span;

      confidence( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        inside
        ? left_disc[ static_cast< size_t >( y ) * left_span + ( x - left_x ) ]
        : 0.0f;
    }
  }

  for( int y = 0; y < height; ++y )
  {
    for( int x = left_x; x < left_x + left_span; ++x )
    {
      auto const value = left_at( x, y );
      auto const partner = x - ( static_cast< int >( value ) >> 4 );

      if( partner < right_x || partner >= right_x + left_span )
      {
        continue;
      }

      auto const agreed =
        std::abs( value + right_at( partner, y ) ) < params.lrc_threshold;

      confidence( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        agreed
        ? std::min(
            left_disc[ static_cast< size_t >( y ) * left_span + ( x - left_x ) ],
            right_disc[ static_cast< size_t >( y ) * left_span +
                        ( partner - right_x ) ] )
        : 0.0f;
    }
  }

  // And the whole map is scaled at the end, after the check rather than inside
  // it, which is what makes the untouched pixels come out on the same scale.
  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      confidence( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) *=
        255.0f;
    }
  }

  // Smooth the disparity weighted by its confidence, and the confidence on its
  // own, then divide -- so a pixel nobody trusts takes its neighbours' value.
  viame::image_of< uint8_t > band( static_cast< size_t >( left_span ),
                                   static_cast< size_t >( height ),
                                   guide.depth() );
  viame::image_of< float > weighted( static_cast< size_t >( left_span ),
                                     static_cast< size_t >( height ), 1 );
  viame::image_of< float > alone( static_cast< size_t >( left_span ),
                                  static_cast< size_t >( height ), 1 );

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < left_span; ++x )
    {
      for( size_t plane = 0; plane < guide.depth(); ++plane )
      {
        band( static_cast< size_t >( x ), static_cast< size_t >( y ), plane ) =
          guide( static_cast< size_t >( left_x + x ),
                 static_cast< size_t >( y ), plane );
      }

      auto const weight = confidence( static_cast< size_t >( left_x + x ),
                                      static_cast< size_t >( y ), 0 );

      alone( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        weight;
      weighted( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) =
        weight * left_at( left_x + x, y );
    }
  }

  auto const smoothed_weighted =
    smooth_globally( band, weighted, params.lambda, params.sigma );
  auto const smoothed_alone =
    smooth_globally( band, alone, params.lambda, params.sigma );

  // OpenCV writes this as `disp_mul_conf.mul( 1 / (conf_filtered + EPS) )`,
  // and both halves of that matter. `EPS` is 1e-43, a **denormal**, so it does
  // not meaningfully bias a real denominator -- it exists to move an exact zero
  // off zero. And `1 / mat` is `cv::divide`, which yields **0** rather than
  // infinity where the divisor is exactly zero: without that the smoothed
  // confidence reaching zero gives an infinity, multiplying to a saturated
  // +-32767 or, where the numerator is zero too, to a NaN. That was the last 64
  // pixels of a 512 by 512 frame, at the full width of the type.
  constexpr float epsilon = 1e-43f;

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < left_span; ++x )
    {
      auto const denominator =
        smoothed_alone( static_cast< size_t >( x ),
                        static_cast< size_t >( y ), 0 ) + epsilon;
      auto const reciprocal = ( denominator != 0.0f )
                              ? 1.0f / denominator : 0.0f;

      out( static_cast< size_t >( left_x + x ), static_cast< size_t >( y ),
           0 ) =
        smoothed_weighted( static_cast< size_t >( x ),
                           static_cast< size_t >( y ), 0 ) * reciprocal;
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif
