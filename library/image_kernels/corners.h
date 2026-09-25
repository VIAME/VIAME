/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Sub-pixel corner refinement
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

#include <image_kernels/warp.h>

#include <limits>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
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

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_CORNERS_H
