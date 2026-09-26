/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Corner detection and sub-pixel refinement
///
/// What `cv::cornerSubPix` did. The idea is one observation: at a corner, the
/// image gradient in the neighbourhood is orthogonal to the vector from the
/// corner to the pixel that carries it. A point on an edge has a gradient
/// across the edge and no displacement along it; a point in flat ground has
/// no gradient at all. So the corner is the point q for which
///
///     sum over the window of  g g^T ( p - q )  =  0
///
/// which is a two by two linear system in q, solved and re-centred until it
/// stops moving.
///
/// The details are OpenCV's, because the calibration's corner positions --
/// and therefore its focal lengths -- follow from them:
///
/// * the window is sampled **bilinearly** about the current estimate, so the
///   refinement is not quantised to the pixel grid it started on;
/// * the gradients are central differences over that sampled window;
/// * each is weighted by a separable mask that falls off from the centre,
///   which keeps a distant edge from dragging the answer;
/// * the iteration stops on either the step size or the count, whichever
///   comes first, as a `TermCriteria` with both does.

#ifndef VIAME_IMAGE_KERNELS_CORNERS_H
#define VIAME_IMAGE_KERNELS_CORNERS_H

#include <image_kernels/filter.h>
#include <image_kernels/warp.h>

#include <limits>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace viame {
namespace image_kernels {

namespace detail {

/// The separable fall-off mask `cv::cornerSubPix` weights its window with.
///
/// A **Gaussian**, `exp( -x^2 )` with x the offset from the centre divided by
/// the half window, so it runs over [-1, 1] whatever the window size. OpenCV
/// builds the same expression in each axis and multiplies the two.
///
/// It is worth being exact about: a quadratic fall-off of roughly the same
/// shape moves a refined corner by 0.003 of a pixel, which is nothing on its
/// own and is a pixel and a half of focal length once a calibration has
/// compounded it over forty corners and twelve views.
inline std::vector< double >
corner_mask( int half )
{
  std::vector< double > axis( 2 * half + 1 );

  for( int i = -half; i <= half; ++i )
  {
    auto const x = static_cast< double >( i ) / static_cast< double >( half );
    axis[ static_cast< size_t >( i + half ) ] = std::exp( -x * x );
  }

  return axis;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Refine each corner to sub-pixel accuracy, in place.
///
/// @param image the single plane the corners were found in
/// @param corners the positions to refine, as x, y
/// @param half_width half the search window, so 5 means an 11 pixel window
/// @param half_height likewise
/// @param iterations the most passes any one corner takes
/// @param epsilon the step size below which a corner is settled
template < typename T >
void
corner_subpix( viame::image_of< T > const& image,
               std::vector< std::pair< double, double > >& corners,
               int half_width = 5, int half_height = 5,
               int iterations = 40, double epsilon = 0.001 )
{
  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "corner_subpix takes a single plane" );
  }

  if( half_width < 1 || half_height < 1 )
  {
    throw std::invalid_argument( "corner_subpix wants a window of at least 1" );
  }

  auto const mask_x = detail::corner_mask( half_width );
  auto const mask_y = detail::corner_mask( half_height );

  // One pixel of margin each way, because the gradients are central
  // differences over the sampled window.
  auto const width = 2 * half_width + 3;
  auto const height = 2 * half_height + 3;

  std::vector< double > window(
    static_cast< size_t >( width ) * static_cast< size_t >( height ) );

  auto const squared_epsilon = epsilon * epsilon;

  for( auto& corner : corners )
  {
    auto current = corner;

    for( int pass = 0; pass < iterations; ++pass )
    {
      // Sample the neighbourhood about where the corner is *now*
      for( int j = 0; j < height; ++j )
      {
        for( int i = 0; i < width; ++i )
        {
          auto const x = current.first + ( i - half_width - 1 );
          auto const y = current.second + ( j - half_height - 1 );

          window[ static_cast< size_t >( j ) * width + i ] =
            sample_bilinear( image, x, y, 0, border_mode::REPLICATE );
        }
      }

      double a = 0.0;
      double b = 0.0;
      double c = 0.0;
      double bb1 = 0.0;
      double bb2 = 0.0;

      for( int j = 1; j < height - 1; ++j )
      {
        for( int i = 1; i < width - 1; ++i )
        {
          auto const at = static_cast< size_t >( j ) * width + i;

          auto const gx = ( window[ at + 1 ] - window[ at - 1 ] ) / 2.0;
          auto const gy = ( window[ at + width ] - window[ at - width ] ) / 2.0;

          auto const weight =
            mask_x[ static_cast< size_t >( i - 1 ) ] *
            mask_y[ static_cast< size_t >( j - 1 ) ];

          auto const gxx = gx * gx * weight;
          auto const gxy = gx * gy * weight;
          auto const gyy = gy * gy * weight;

          auto const px = static_cast< double >( i - half_width - 1 );
          auto const py = static_cast< double >( j - half_height - 1 );

          a += gxx;
          b += gxy;
          c += gyy;
          bb1 += gxx * px + gxy * py;
          bb2 += gxy * px + gyy * py;
        }
      }

      auto const determinant = a * c - b * b;

      // A flat or purely linear neighbourhood has no corner in it; leaving
      // the estimate where it is beats moving it by a divided-by-zero.
      if( std::abs( determinant ) <= std::numeric_limits< double >::epsilon() *
                                     std::abs( a + c ) )
      {
        break;
      }

      auto const dx = ( c * bb1 - b * bb2 ) / determinant;
      auto const dy = ( a * bb2 - b * bb1 ) / determinant;

      std::pair< double, double > const next{ current.first + dx,
                                              current.second + dy };

      auto const moved = ( next.first - current.first ) *
                         ( next.first - current.first ) +
                         ( next.second - current.second ) *
                         ( next.second - current.second );

      current = next;

      // A corner that has wandered outside the image is not a corner
      if( current.first < 0.0 || current.second < 0.0 ||
          current.first >= static_cast< double >( image.width() ) ||
          current.second >= static_cast< double >( image.height() ) )
      {
        current = corner;
        break;
      }

      if( moved < squared_epsilon )
      {
        break;
      }
    }

    corner = current;
  }
}


// ----------------------------------------------------------------------------
/// The smaller eigenvalue of the structure tensor at every pixel.
///
/// What `cv::cornerMinEigenVal` computed, and the corner strength
/// `goodFeaturesToTrack` ranks by: the gradient covariance summed over a
/// block, whose smaller eigenvalue is large only where the gradient points
/// in two directions at once -- which is what a corner is and an edge is
/// not.
///
/// The scaling is OpenCV's and matters, because the quality threshold below
/// is a fraction of the largest value and a caller comparing against
/// `minEigThreshold` is comparing against an absolute one: the Sobel is
/// divided by `2^(aperture-1) * block_size`, and by 255 more for a byte
/// image, so the answer is in units of the normalised gradient rather than
/// of the pixel.
///
/// @param image one plane
/// @param block_size the side of the block the covariance is summed over
/// @param aperture the Sobel size, which has to be 3 -- OpenCV's 1, 5, 7 and
///        -1 are refused rather than approximated, since nothing asks for
///        them and a silently different gradient would move every corner
template < typename T >
viame::image_of< float >
min_eigen_value( viame::image_of< T > const& image, int block_size = 3,
                 int aperture = 3 )
{
  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "min_eigen_value takes a single plane" );
  }

  if( block_size < 1 || aperture != 3 )
  {
    throw std::invalid_argument(
      "min_eigen_value wants a positive block and a Sobel of 3" );
  }

  auto scale = static_cast< double >( 1 << ( aperture - 1 ) ) * block_size;

  if( std::is_same< T, uint8_t >::value )
  {
    scale *= 255.0;
  }

  scale = 1.0 / scale;

  auto const width = image.width();
  auto const height = image.height();

  viame::image_of< float > out( width, height, 1 );

  if( width == 0 || height == 0 )
  {
    return out;
  }

  // `sobel< float >` would give the same two planes -- the kernels are
  // `sobel_kernel_1d`'s and the sums are still in double, so only the order
  // of two exact additions differs -- but it reaches them through
  // `filter_2d`, which sweeps the **square** nine tap kernel and resolves the
  // border with a switch on each of those taps. Eighteen dispatched calls a
  // pixel for the two derivatives was 0.219 s of the 0.245 it took to find a
  // thousand corners in a 1080p frame.
  //
  // So the derivatives are separable here, and the horizontal halves live in
  // a three row ring rather than in two full planes: the vertical half only
  // ever wants rows y-1, y and y+1, and reflecting at the edge asks for a row
  // that is already one of those three, so nothing has to be kept. Writing
  // them out instead is thirty-two megabytes stored and loaded again per
  // frame, which costs more than the arithmetic does.
  auto const smooth = sobel_kernel_1d( 0, static_cast< size_t >( aperture ) );
  auto const differ = sobel_kernel_1d( 1, static_cast< size_t >( aperture ) );

  // In float, because `cv::cornerMinEigenVal` is: its Sobel is CV_32F and its
  // box filter accumulates in float, so this is not a precision concession to
  // speed but the same precision OpenCV chose, and it halves both the traffic
  // and the width of a vector lane.
  std::vector< float > narrow_smooth( smooth.begin(), smooth.end() );
  std::vector< float > narrow_differ( differ.begin(), differ.end() );
  auto const narrow_scale = static_cast< float >( scale );

  std::vector< float > ring_differ( 3 * width ), ring_smooth( 3 * width );
  long filled = -1;

  // `image_of::operator()` is three multiplications to reach a pixel, and the
  // row's pixels are adjacent, so the row is found once and walked. For the
  // packed single plane case -- which is every caller -- the walk is a plain
  // stride of one and the compiler can see it.
  auto const* const base = image.first_pixel();
  auto const across_step = image.w_step();
  auto const down_step = image.h_step();
  auto const packed = across_step == 1;

  auto const fill =
    [ & ]( size_t row )
    {
      auto* to_differ = ring_differ.data() + ( row % 3 ) * width;
      auto* to_smooth = ring_smooth.data() + ( row % 3 ) * width;

      auto const* source = base + down_step * static_cast< ptrdiff_t >( row );

      auto const one =
        [ & ]( long i, float& a, float& b )
        {
          for( size_t k = 0; k < 3; ++k )
          {
            auto const index = detail::border_index(
              i + static_cast< long >( k ) - 1, static_cast< long >( width ),
              border_mode::REFLECT_101 );
            auto const value = static_cast< float >(
              source[ across_step * static_cast< ptrdiff_t >( index ) ] );
            a += narrow_differ[ k ] * value;
            b += narrow_smooth[ k ] * value;
          }
        };

      // The interior, where the three taps are simply adjacent
      if( packed )
      {
        for( size_t x = 1; x + 1 < width; ++x )
        {
          float a = 0.0f;
          float b = 0.0f;

          for( size_t k = 0; k < 3; ++k )
          {
            auto const value = static_cast< float >( source[ x + k - 1 ] );
            a += narrow_differ[ k ] * value;
            b += narrow_smooth[ k ] * value;
          }

          to_differ[ x ] = a;
          to_smooth[ x ] = b;
        }
      }
      else
      {
        for( size_t x = 1; x + 1 < width; ++x )
        {
          float a = 0.0f;
          float b = 0.0f;
          one( static_cast< long >( x ), a, b );
          to_differ[ x ] = a;
          to_smooth[ x ] = b;
        }
      }

      // And the two columns whose taps leave the image, which for a width of
      // one is the same column twice and harmlessly so
      for( auto const x : { size_t{ 0 }, width - 1 } )
      {
        float a = 0.0f;
        float b = 0.0f;
        one( static_cast< long >( x ), a, b );
        to_differ[ x ] = a;
        to_smooth[ x ] = b;
      }
    };

  // The three distinct entries of the gradient covariance, before the block
  auto const radius = block_size / 2;

  std::vector< float > xx( width * height ), xy( width * height ),
                       yy( width * height );

  for( size_t y = 0; y < height; ++y )
  {
    auto const wanted = std::min( y + 1, height - 1 );

    while( filled < static_cast< long >( wanted ) )
    {
      fill( static_cast< size_t >( ++filled ) );
    }

    size_t rows[ 3 ];

    for( size_t k = 0; k < 3; ++k )
    {
      rows[ k ] = static_cast< size_t >( detail::border_index(
        static_cast< long >( y ) + static_cast< long >( k ) - 1,
        static_cast< long >( height ), border_mode::REFLECT_101 ) ) % 3;
    }

    auto* to_xx = xx.data() + y * width;
    auto* to_xy = xy.data() + y * width;
    auto* to_yy = yy.data() + y * width;

    for( size_t x = 0; x < width; ++x )
    {
      float across = 0.0f;
      float down = 0.0f;

      for( size_t k = 0; k < 3; ++k )
      {
        across += narrow_smooth[ k ] * ring_differ[ rows[ k ] * width + x ];
        down += narrow_differ[ k ] * ring_smooth[ rows[ k ] * width + x ];
      }

      auto const gx = across * narrow_scale;
      auto const gy = down * narrow_scale;

      to_xx[ x ] = gx * gx;
      to_xy[ x ] = gx * gy;
      to_yy[ x ] = gy * gy;
    }
  }

  // An **unnormalised** box sum of all three at once, which is what
  // `boxFilter` with `normalize` off gives and what the halving below is
  // paired with. All three share the sweep because they share every index:
  // three separate calls made six passes over twenty-four megabytes where
  // this makes two.
  std::vector< float > across_xx( width * height ), across_xy( width * height ),
                       across_yy( width * height );

  for( size_t y = 0; y < height; ++y )
  {
    auto const* from_xx = xx.data() + y * width;
    auto const* from_xy = xy.data() + y * width;
    auto const* from_yy = yy.data() + y * width;

    auto* to_xx = across_xx.data() + y * width;
    auto* to_xy = across_xy.data() + y * width;
    auto* to_yy = across_yy.data() + y * width;

    for( size_t x = 0; x < width; ++x )
    {
      float a = 0.0f, b = 0.0f, c = 0.0f;

      // The taps are added in the same order either way, so the answer is the
      // same to the last bit; the split keeps the border resolution off the
      // pixels that are nowhere near one
      if( x >= static_cast< size_t >( radius ) &&
          x + static_cast< size_t >( radius ) < width )
      {
        for( int k = -radius; k <= radius; ++k )
        {
          auto const at = x + static_cast< size_t >( k );
          a += from_xx[ at ];
          b += from_xy[ at ];
          c += from_yy[ at ];
        }
      }
      else
      {
        for( int k = -radius; k <= radius; ++k )
        {
          auto const at = static_cast< size_t >( detail::border_index(
            static_cast< long >( x ) + k, static_cast< long >( width ),
            border_mode::REFLECT_101 ) );
          a += from_xx[ at ];
          b += from_xy[ at ];
          c += from_yy[ at ];
        }
      }

      to_xx[ x ] = a;
      to_xy[ x ] = b;
      to_yy[ x ] = c;
    }
  }

  std::vector< float const* > rows_xx, rows_xy, rows_yy;
  rows_xx.reserve( static_cast< size_t >( 2 * radius + 1 ) );
  rows_xy.reserve( static_cast< size_t >( 2 * radius + 1 ) );
  rows_yy.reserve( static_cast< size_t >( 2 * radius + 1 ) );

  for( size_t y = 0; y < height; ++y )
  {
    rows_xx.clear();
    rows_xy.clear();
    rows_yy.clear();

    for( int k = -radius; k <= radius; ++k )
    {
      auto const at = static_cast< size_t >( detail::border_index(
        static_cast< long >( y ) + k, static_cast< long >( height ),
        border_mode::REFLECT_101 ) );
      rows_xx.push_back( across_xx.data() + at * width );
      rows_xy.push_back( across_xy.data() + at * width );
      rows_yy.push_back( across_yy.data() + at * width );
    }

    auto* destination = out.first_pixel() +
      out.h_step() * static_cast< ptrdiff_t >( y );

    for( size_t x = 0; x < width; ++x )
    {
      float first = 0.0f, cross = 0.0f, second = 0.0f;

      for( size_t k = 0; k < rows_xx.size(); ++k )
      {
        first += rows_xx[ k ][ x ];
        cross += rows_xy[ k ][ x ];
        second += rows_yy[ k ][ x ];
      }

      auto const a = first * 0.5f;
      auto const b = cross;
      auto const c = second * 0.5f;

      destination[ out.w_step() * static_cast< ptrdiff_t >( x ) ] =
        a + c - std::sqrt( ( a - c ) * ( a - c ) + b * b );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// The corners `cv::goodFeaturesToTrack` would have found, strongest first.
///
/// Shi and Tomasi's measure -- `min_eigen_value` above -- thresholded at
/// \p quality_level of the strongest in the image, reduced to the local
/// maxima of a three by three window, sorted, and then thinned so that no
/// two kept corners are within \p min_distance of each other.
///
/// Two details are OpenCV's and are what make the list the same list:
///
/// * the one pixel rim is never a corner, whatever its strength, because the
///   maximum it is compared against would have to read outside the image;
/// * equal strengths are broken by position, in row major order, because
///   OpenCV sorts pointers into the strength image and a stable sort on
///   equal values leaves them in the order they were collected.
///
/// @param image one plane
/// @param max_corners the most to return, or zero for no limit
/// @param quality_level the fraction of the strongest a corner must reach
/// @param min_distance how far apart two corners have to be
/// @param block_size the block the corner measure sums over
/// @param aperture the Sobel size
template < typename T >
std::vector< std::pair< float, float > >
good_features_to_track( viame::image_of< T > const& image,
                        int max_corners = 1000, double quality_level = 0.01,
                        double min_distance = 10.0, int block_size = 3,
                        int aperture = 3 )
{
  if( quality_level <= 0.0 || min_distance < 0.0 )
  {
    throw std::invalid_argument(
      "good_features_to_track wants a positive quality and distance" );
  }

  auto const strength = min_eigen_value( image, block_size, aperture );

  auto const width = strength.width();
  auto const height = strength.height();

  std::vector< std::pair< float, float > > out;

  if( width < 3 || height < 3 )
  {
    return out;
  }

  float best = strength( 0, 0, 0 );

  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      best = std::max( best, strength( x, y, 0 ) );
    }
  }

  auto const floor_value = static_cast< float >( best * quality_level );

  // Position and strength of every local maximum that clears the threshold,
  // collected in row major order so that a stable sort keeps OpenCV's
  // tie-break
  struct candidate { float value; size_t x; size_t y; };
  std::vector< candidate > found;

  for( size_t y = 1; y + 1 < height; ++y )
  {
    for( size_t x = 1; x + 1 < width; ++x )
    {
      auto const value = strength( x, y, 0 );

      if( !( value > floor_value ) || value == 0.0f )
      {
        continue;
      }

      auto highest = value;

      for( size_t j = y - 1; j <= y + 1; ++j )
      {
        for( size_t i = x - 1; i <= x + 1; ++i )
        {
          highest = std::max( highest, strength( i, j, 0 ) );
        }
      }

      if( value == highest )
      {
        found.push_back( { value, x, y } );
      }
    }
  }

  std::stable_sort( found.begin(), found.end(),
                    []( candidate const& a, candidate const& b )
                    { return a.value > b.value; } );

  auto const wanted = static_cast< size_t >(
    max_corners > 0 ? max_corners : static_cast< int >( found.size() ) );

  if( min_distance < 1.0 )
  {
    for( auto const& one : found )
    {
      out.emplace_back( static_cast< float >( one.x ),
                        static_cast< float >( one.y ) );

      if( out.size() == wanted ) { break; }
    }

    return out;
  }

  // The grid is OpenCV's way of asking only the neighbours: a cell is a
  // minimum distance across, so nothing outside the nine cells around a
  // candidate can be too close to it.
  auto const cell = static_cast< size_t >( std::max(
    1L, static_cast< long >( std::nearbyint( min_distance ) ) ) );
  auto const across = ( width + cell - 1 ) / cell;
  auto const down = ( height + cell - 1 ) / cell;

  std::vector< std::vector< std::pair< float, float > > > grid( across * down );

  auto const squared = min_distance * min_distance;

  for( auto const& one : found )
  {
    auto const cx = one.x / cell;
    auto const cy = one.y / cell;

    auto keep = true;

    auto const low_x = cx > 0 ? cx - 1 : 0;
    auto const low_y = cy > 0 ? cy - 1 : 0;
    auto const high_x = std::min( cx + 1, across - 1 );
    auto const high_y = std::min( cy + 1, down - 1 );

    for( auto j = low_y; j <= high_y && keep; ++j )
    {
      for( auto i = low_x; i <= high_x && keep; ++i )
      {
        for( auto const& other : grid[ j * across + i ] )
        {
          auto const dx = static_cast< double >( one.x ) - other.first;
          auto const dy = static_cast< double >( one.y ) - other.second;

          if( dx * dx + dy * dy < squared )
          {
            keep = false;
            break;
          }
        }
      }
    }

    if( !keep ) { continue; }

    std::pair< float, float > const corner{ static_cast< float >( one.x ),
                                            static_cast< float >( one.y ) };

    grid[ cy * across + cx ].push_back( corner );
    out.push_back( corner );

    if( out.size() == wanted ) { break; }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_CORNERS_H
