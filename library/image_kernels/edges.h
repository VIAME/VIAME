/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Canny edge detection
///
/// What `cv::Canny` does, reproduced rather than approximated: the goldens
/// for `hough_circle` are recordings of cv2 compared at a tolerance of zero,
/// and the circle transform runs Canny itself, so an edge map that is merely
/// similar moves every circle that follows it.
///
/// Three things carry the agreement, and none is the obvious reading:
///
/// * the gradient is a **16 bit** Sobel with a replicated border, not a
///   floating point one;
/// * the direction is quantised by comparing `|dy| << 15` against
///   `|dx| * 13573` -- `tan(22.5)` in 15 bits -- rather than by taking an
///   angle, so the eight sectors fall out of integer comparisons;
/// * the suppression is **asymmetric**: strictly greater than the neighbour
///   behind, greater or equal to the one ahead.

#ifndef VIAME_IMAGE_KERNELS_EDGES_H
#define VIAME_IMAGE_KERNELS_EDGES_H

#include <image_kernels/color.h>
#include <image_kernels/filter.h>
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

namespace detail {

/// `tan(22.5 degrees)` in 15 fractional bits, which is OpenCV's `TG22`.
constexpr int canny_shift = 15;
constexpr int canny_tg22 = 13573;

/// One separable Sobel pass into a signed integer plane, as `cv::Sobel` with
/// `CV_16S` and a replicated border gives it.
///
/// The kernels are integers and the input is an integer, so there is nothing
/// to round: the only arithmetic OpenCV does that this has to copy is the
/// saturation into 16 bits.
inline std::vector< int >
sobel_plane( viame::image_of< uint8_t > const& image, int order_x,
             int order_y, size_t aperture )
{
  auto const across = sobel_kernel_1d( order_x, aperture );
  auto const down = sobel_kernel_1d( order_y, aperture );

  auto const width = image.width();
  auto const height = image.height();
  auto const anchor = static_cast< long >( across.size() / 2 );

  std::vector< double > mid( width * height, 0.0 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      auto total = 0.0;

      for( size_t k = 0; k < across.size(); ++k )
      {
        total += across[ k ] * sample_with_border(
          image, static_cast< long >( i ) + static_cast< long >( k ) - anchor,
          static_cast< long >( j ), 0, border_mode::REPLICATE );
      }

      mid[ j * width + i ] = total;
    }
  }

  std::vector< int > out( width * height, 0 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      auto total = 0.0;

      for( size_t k = 0; k < down.size(); ++k )
      {
        auto const row = border_index(
          static_cast< long >( j ) + static_cast< long >( k ) - anchor,
          static_cast< long >( height ), border_mode::REPLICATE );

        total += down[ k ] * mid[ static_cast< size_t >( row ) * width + i ];
      }

      auto const rounded = static_cast< long >( std::nearbyint( total ) );

      out[ j * width + i ] = static_cast< int >(
        std::min< long >( std::max< long >( rounded, -32768 ), 32767 ) );
    }
  }

  return out;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::Canny`: the edges of \p image as 0 or 255.
///
/// \p low and \p high are the hysteresis thresholds, in the same units as the
/// gradient: with \p l2_gradient they are compared against the squared
/// magnitude, which is why OpenCV clamps them to 32767 and squares them
/// before flooring rather than after.
inline viame::image_of< uint8_t >
canny( viame::image_of< uint8_t > const& image, double low, double high,
       size_t aperture = 3, bool l2_gradient = false )
{
  // `require_planes` means "at least", which is the wrong test here: with
  // three planes it would pass and this would quietly read plane 0, where
  // `cv::Canny` takes the strongest channel of the three per pixel. Refused
  // rather than half-implemented -- a caller wanting the OpenCV behaviour
  // should say which channel it meant, or convert first, as the circle
  // detector does.
  if( image.depth() != 1 )
  {
    throw std::invalid_argument(
      "canny needs a single plane, got " + std::to_string( image.depth() ) +
      "; cv2 takes the strongest channel and this does not" );
  }

  // 7 is refused rather than approximated. `cv::Canny` accepts it, and at
  // that size the 16 bit gradient **saturates** -- a 7-tap Sobel reaches
  // 326400 where the type stops at 32767 -- after which cv2's answer is not
  // the one this suppression produces from a saturated gradient. Measured: fed
  // cv2's *own* `Sobel` output at aperture 7 the edge map still differs on 113
  // pixels of 2240 in L1 and 154 in L2, identically to feeding it ours, so the
  // difference is inside cv2's Canny and not in the derivative. Apertures 3
  // and 5 are identical over 160 configurations, and `hough_circles` uses 3.
  if( aperture != 3 && aperture != 5 )
  {
    throw std::invalid_argument(
      "canny: the aperture is 3 or 5; 7 saturates the 16 bit gradient" );
  }

  if( low > high )
  {
    std::swap( low, high );
  }

  auto const width = image.width();
  auto const height = image.height();

  viame::image_of< uint8_t > out( width, height, 1 );

  if( width == 0 || height == 0 )
  {
    return out;
  }

  auto const dx = detail::sobel_plane( image, 1, 0, aperture );
  auto const dy = detail::sobel_plane( image, 0, 1, aperture );

  if( l2_gradient )
  {
    low = std::min( 32767.0, low );
    high = std::min( 32767.0, high );

    if( low > 0.0 ) { low *= low; }
    if( high > 0.0 ) { high *= high; }
  }

  auto const low_at = static_cast< long >( std::floor( low ) );
  auto const high_at = static_cast< long >( std::floor( high ) );

  std::vector< long > magnitude( width * height );

  for( size_t at = 0; at < magnitude.size(); ++at )
  {
    auto const x = static_cast< long >( dx[ at ] );
    auto const y = static_cast< long >( dy[ at ] );

    magnitude[ at ] = l2_gradient ? x * x + y * y
                                  : std::abs( x ) + std::abs( y );
  }

  // 0 is a candidate the flood may still claim, 1 is decided against, 2 is an
  // edge. The frame of 1s means the flood below never needs a bounds test.
  auto const stride = width + 2;
  std::vector< uint8_t > map( stride * ( height + 2 ), 1 );
  std::vector< size_t > pending;

  auto const at = [ & ]( size_t i, size_t j ) { return ( j + 1 ) * stride + i + 1; };
  auto const reach = [ & ]( long i, long j ) -> long
  {
    if( i < 0 || j < 0 || i >= static_cast< long >( width ) ||
        j >= static_cast< long >( height ) )
    {
      return 0;
    }

    return magnitude[ static_cast< size_t >( j ) * width +
                      static_cast< size_t >( i ) ];
  };

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      auto const here = magnitude[ j * width + i ];

      if( here <= low_at )
      {
        continue;
      }

      auto const gx = static_cast< long >( dx[ j * width + i ] );
      auto const gy = static_cast< long >( dy[ j * width + i ] );
      auto const x = std::abs( gx );
      auto const y = std::abs( gy ) << detail::canny_shift;
      auto const tg22x = x * detail::canny_tg22;

      auto const left = static_cast< long >( i ) - 1;
      auto const right = static_cast< long >( i ) + 1;
      auto const up = static_cast< long >( j ) - 1;
      auto const down = static_cast< long >( j ) + 1;

      bool peak = false;

      if( y < tg22x )
      {
        // Within 22.5 degrees of horizontal: compare across.
        peak = here > reach( left, static_cast< long >( j ) ) &&
               here >= reach( right, static_cast< long >( j ) );
      }
      else if( y > tg22x + ( x << ( detail::canny_shift + 1 ) ) )
      {
        // Within 22.5 degrees of vertical: compare down the column.
        peak = here > reach( static_cast< long >( i ), up ) &&
               here >= reach( static_cast< long >( i ), down );
      }
      else
      {
        // Diagonal, and which diagonal is the sign of the product.
        auto const step = ( ( gx ^ gy ) < 0 ) ? -1 : 1;

        peak = here > reach( static_cast< long >( i ) - step, up ) &&
               here > reach( static_cast< long >( i ) + step, down );
      }

      if( !peak )
      {
        continue;
      }

      if( here > high_at )
      {
        map[ at( i, j ) ] = 2;
        pending.push_back( at( i, j ) );
      }
      else
      {
        map[ at( i, j ) ] = 0;
      }
    }
  }

  // Hysteresis. OpenCV also declines to seed a pixel whose left neighbour was
  // just seeded, or whose upstairs neighbour is already an edge; both are
  // skipped here because they cannot change the answer -- the pixel is left a
  // candidate and is adjacent to whatever caused the skip, so this flood
  // reaches it regardless.
  while( !pending.empty() )
  {
    auto const here = pending.back();
    pending.pop_back();

    long const around[ 8 ] = { -static_cast< long >( stride ) - 1,
                               -static_cast< long >( stride ),
                               -static_cast< long >( stride ) + 1,
                               -1, 1,
                               static_cast< long >( stride ) - 1,
                               static_cast< long >( stride ),
                               static_cast< long >( stride ) + 1 };

    for( auto const step : around )
    {
      auto const next = static_cast< size_t >(
        static_cast< long >( here ) + step );

      if( map[ next ] == 0 )
      {
        map[ next ] = 2;
        pending.push_back( next );
      }
    }
  }

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      out( i, j, 0 ) = ( map[ at( i, j ) ] == 2 ) ? 255 : 0;
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif
