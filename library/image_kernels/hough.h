/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The Hough gradient circle transform
///
/// `cv::HoughCircles` under `HOUGH_GRADIENT`, reproduced rather than
/// approximated. It runs `canny` itself, so `edges.h` has to be exact first.
///
/// Two things decide whether this agrees, and neither is in the algorithm as
/// it is usually described:
///
/// * the value the output is **sorted by** is the count the radius histogram
///   settled on, not the accumulator peak at the centre. OpenCV builds
///   `EstimatedCircle(Vec3f(...), maxCount)` and `cmpAccum` reads that field.
///   Sorting by the peak gives nearly the same circles in nearly the same
///   order, and picks a different member of every `min_dist` cluster;
/// * the whole thing runs in **single precision**. `idp`, the gradient
///   magnitude, the vote steps, the centres, the radii and the overlap test
///   are all float there, and in double the answers part company on scenes
///   with enough candidates to have a tail.

#ifndef VIAME_IMAGE_KERNELS_HOUGH_H
#define VIAME_IMAGE_KERNELS_HOUGH_H

#include <image_kernels/edges.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

/// A circle as `cv::HoughCircles` reports one.
struct circle
{
  float x = 0.0f;
  float y = 0.0f;
  float radius = 0.0f;
};

namespace detail {

/// The accumulator position of a peak, with its value, for `hough_cmp_gt`.
struct hough_peak
{
  int index = 0;
  int votes = 0;
  int x = 0;
  int y = 0;
};

/// What the radius stage produced, in the shape `cmpAccum` orders.
struct estimated_circle
{
  int count = 0;
  float radius = 0.0f;
  float x = 0.0f;
  float y = 0.0f;
};

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::HoughCircles` with `HOUGH_GRADIENT`.
///
/// \p canny_threshold is OpenCV's `param1`, the high hysteresis threshold, and
/// the low one is half of it. \p acc_threshold is `param2`.
inline std::vector< circle >
hough_circles( viame::image_of< uint8_t > const& image, double dp,
               double min_dist, double canny_threshold, double acc_threshold,
               int min_radius, int max_radius, int max_circles = -1 )
{
  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "hough_circles needs a single plane" );
  }

  if( min_dist <= 0.0 )
  {
    throw std::invalid_argument( "hough_circles: min_dist has to be positive" );
  }

  constexpr int shift = 10;
  constexpr int one = 1 << shift;
  constexpr int bins_per_dr = 10;

  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );

  std::vector< circle > out;

  if( width == 0 || height == 0 )
  {
    return out;
  }

  // OpenCV's radius conventions, which the shipped pipeline depends on: it
  // sets both radii to 0, and a `max_radius` of 0 there means the larger image
  // extent rather than "no radii at all". Anything above that extent is
  // clamped to it. Getting this wrong cost seven applet tests, which run the
  // circles pipeline at its shipped settings and got no detections at all,
  // while the golden -- which sets 3 and 20 explicitly -- passed throughout.
  auto const limit = std::max( width, height );

  if( max_radius < 0 )
  {
    // `cv::HoughCircles` reads a negative maximum as "centres only, radius
    // zero", which is a debugging affordance rather than a detector setting.
    // Refused rather than silently treated as the clamp above.
    throw std::invalid_argument(
      "hough_circles: a negative max_radius is cv2's centres-only mode, "
      "which this does not implement" );
  }

  // Only a zero maximum defaults. A value above the image extent is **not**
  // clamped down, which one scene suggested and four others disproved: the
  // vote ray's length and the radius histogram's bin count both depend on
  // `max_radius`, so 300 on a 100 by 120 image is not 120 and finds a circle
  // that 120 does not.
  if( max_radius == 0 )
  {
    max_radius = limit;
  }

  min_radius = std::max( min_radius, 0 );

  if( max_radius < min_radius )
  {
    return out;
  }

  auto const scale = ( dp < 1.0 ) ? 1.0f : static_cast< float >( dp );
  auto const inverse = 1.0f / scale;

  auto const edges = canny(
    image, std::max( canny_threshold / 2.0, 1.0 ), canny_threshold, 3, false );

  auto const dx = detail::sobel_plane( image, 1, 0, 3 );
  auto const dy = detail::sobel_plane( image, 0, 1, 3 );

  auto const acols = static_cast< int >(
    std::ceil( static_cast< float >( width ) * inverse ) );
  auto const arows = static_cast< int >(
    std::ceil( static_cast< float >( height ) * inverse ) );

  std::vector< int > accumulator(
    static_cast< size_t >( arows + 2 ) * ( acols + 2 ), 0 );
  auto const astep = acols + 2;

  std::vector< float > nz_x;
  std::vector< float > nz_y;

  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      if( edges( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 ) == 0 )
      {
        continue;
      }

      auto const at = static_cast< size_t >( y ) * image.width() + x;
      auto const vx = static_cast< float >( dx[ at ] );
      auto const vy = static_cast< float >( dy[ at ] );

      if( vx == 0.0f && vy == 0.0f )
      {
        continue;
      }

      auto const magnitude = std::sqrt( vx * vx + vy * vy );

      auto sx = static_cast< int >( std::nearbyint(
        vx * inverse * static_cast< float >( one ) / magnitude ) );
      auto sy = static_cast< int >( std::nearbyint(
        vy * inverse * static_cast< float >( one ) / magnitude ) );

      auto const x0 = static_cast< int >( std::nearbyint(
        static_cast< float >( x ) * inverse * static_cast< float >( one ) ) );
      auto const y0 = static_cast< int >( std::nearbyint(
        static_cast< float >( y ) * inverse * static_cast< float >( one ) ) );

      // Both ways along the gradient, which is what makes a circle's edge
      // vote for its own centre whichever side of it the edge is on.
      for( int pass = 0; pass < 2; ++pass )
      {
        auto x1 = x0 + min_radius * sx;
        auto y1 = y0 + min_radius * sy;

        for( int r = min_radius; r <= max_radius; ++r, x1 += sx, y1 += sy )
        {
          auto const x2 = x1 >> shift;
          auto const y2 = y1 >> shift;

          if( x2 < 0 || x2 >= acols || y2 < 0 || y2 >= arows )
          {
            break;
          }

          ++accumulator[ static_cast< size_t >( y2 + 1 ) * astep + x2 + 1 ];
        }

        sx = -sx;
        sy = -sy;
      }

      nz_x.push_back( static_cast< float >( x ) );
      nz_y.push_back( static_cast< float >( y ) );
    }
  }

  if( nz_x.empty() )
  {
    return out;
  }

  // The peak test is asymmetric -- strictly greater behind, greater or equal
  // ahead -- and the sweep skips the first cell of each axis while keeping the
  // last, which falls out of OpenCV walking a padded accumulator whose padding
  // is all at the far end.
  std::vector< detail::hough_peak > peaks;

  for( int y = 2; y < arows + 1; ++y )
  {
    for( int x = 2; x < acols + 1; ++x )
    {
      auto const here = accumulator[ static_cast< size_t >( y ) * astep + x ];

      if( here > acc_threshold &&
          here > accumulator[ static_cast< size_t >( y ) * astep + x - 1 ] &&
          here >= accumulator[ static_cast< size_t >( y ) * astep + x + 1 ] &&
          here > accumulator[ static_cast< size_t >( y - 1 ) * astep + x ] &&
          here >= accumulator[ static_cast< size_t >( y + 1 ) * astep + x ] )
      {
        peaks.push_back( detail::hough_peak{ y * astep + x, here, x, y } );
      }
    }
  }

  // `hough_cmp_gt`: by votes descending, ties by position in raster order.
  std::sort( peaks.begin(), peaks.end(),
             []( detail::hough_peak const& left, detail::hough_peak const& right )
             {
               return left.votes != right.votes ? left.votes > right.votes
                                                : left.index < right.index;
             } );

  auto const dr = scale;
  auto const smallest = static_cast< float >( min_radius ) *
                        static_cast< float >( min_radius );
  auto const largest = static_cast< float >( max_radius ) *
                       static_cast< float >( max_radius );
  auto const bin_count = static_cast< int >( std::nearbyint(
    static_cast< float >( max_radius - min_radius ) / dr * bins_per_dr ) );

  std::vector< detail::estimated_circle > estimated;
  std::vector< int > bins;

  for( auto const& peak : peaks )
  {
    if( bin_count <= 0 )
    {
      continue;
    }

    auto const cx = ( static_cast< float >( peak.x - 1 ) + 0.5f ) * scale;
    auto const cy = ( static_cast< float >( peak.y - 1 ) + 0.5f ) * scale;

    bins.assign( static_cast< size_t >( bin_count ), 0 );

    auto any = false;

    for( size_t k = 0; k < nz_x.size(); ++k )
    {
      auto const ddx = cx - nz_x[ k ];
      auto const ddy = cy - nz_y[ k ];
      auto const distance = ddx * ddx + ddy * ddy;

      if( distance < smallest || distance > largest )
      {
        continue;
      }

      any = true;

      auto const at = static_cast< int >( std::nearbyint(
        ( std::sqrt( distance ) - static_cast< float >( min_radius ) ) / dr *
        static_cast< float >( bins_per_dr ) ) );

      ++bins[ static_cast< size_t >(
        std::min( std::max( at, 0 ), bin_count - 1 ) ) ];
    }

    if( !any )
    {
      continue;
    }

    // Swept from the top, a window of `bins_per_dr` bins at a time, keeping
    // the window whose count weighted against its own radius is best. The
    // outer decrement fires whether or not the inner loop ran, and is worth
    // most of a pixel on its own.
    auto best = 0.0f;
    auto most = 0;

    for( int j = bin_count - 1; j > 0; --j )
    {
      if( bins[ static_cast< size_t >( j ) ] == 0 )
      {
        continue;
      }

      auto const top = j;
      auto current = 0;

      while( j > top - bins_per_dr && j >= 0 )
      {
        current += bins[ static_cast< size_t >( j ) ];
        --j;
      }

      auto const radius = static_cast< float >( top + j ) / 2.0f /
                          static_cast< float >( bins_per_dr ) * dr +
                          static_cast< float >( min_radius );

      if( static_cast< float >( current ) * best >=
          static_cast< float >( most ) * radius ||
          ( best < std::numeric_limits< float >::epsilon() && current >= most ) )
      {
        best = radius;
        most = current;
      }
    }

    if( most > acc_threshold )
    {
      estimated.push_back(
        detail::estimated_circle{ most, best, cx, cy } );
    }
  }

  // `cmpAccum`, and the count is the histogram's rather than the peak's.
  std::sort( estimated.begin(), estimated.end(),
             []( detail::estimated_circle const& left,
                 detail::estimated_circle const& right )
             {
               if( left.count != right.count ) { return left.count > right.count; }
               if( left.radius != right.radius ) { return left.radius > right.radius; }
               if( left.x != right.x ) { return left.x < right.x; }
               return left.y < right.y;
             } );

  auto const apart = static_cast< float >( min_dist ) *
                     static_cast< float >( min_dist );

  for( auto const& candidate : estimated )
  {
    auto keep = true;

    for( auto const& taken : out )
    {
      auto const ddx = candidate.x - taken.x;
      auto const ddy = candidate.y - taken.y;

      if( ddx * ddx + ddy * ddy < apart )
      {
        keep = false;
      }
    }

    if( !keep )
    {
      continue;
    }

    out.push_back( circle{ candidate.x, candidate.y, candidate.radius } );

    if( max_circles > 0 && static_cast< int >( out.size() ) >= max_circles )
    {
      break;
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif
