/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The distance from each pixel to the nearest zero
///
/// What `cv::distanceTransform` did, in the mode every caller in this tree
/// asks for: `DIST_L2` with a mask of 3.
///
/// That mode is **not** the Euclidean distance, and the name is the only
/// thing that suggests it is. It is a chamfer approximation -- two passes
/// over the image, each taking the best of a handful of neighbours plus a
/// fixed step cost -- and it is within a couple of per cent of Euclidean
/// while costing two sweeps instead of a search. OpenCV's step costs for the
/// three by three mask are 0.955 sideways and 1.3693 diagonally, which are
/// not 1 and sqrt(2): they are Borgefors' values, fitted to minimise the
/// worst error of the approximation rather than to be exact along an axis.
/// Using 1 and sqrt(2) instead gives a transform that is exact on a
/// horizontal run and 8% long on a diagonal one.
///
/// A caller who wants true Euclidean distance wants a different algorithm --
/// Felzenszwalb's, which is also two passes but over the squared distance --
/// and not different constants here. `DIST_MASK_PRECISE` is OpenCV's name for
/// that one, and nothing in this tree uses it.

#ifndef VIAME_IMAGE_KERNELS_DISTANCE_H
#define VIAME_IMAGE_KERNELS_DISTANCE_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace viame {
namespace image_kernels {

/// Borgefors' step costs for the three by three chamfer, as OpenCV uses them.
constexpr float chamfer_orthogonal = 0.955f;
constexpr float chamfer_diagonal = 1.3693f;

// ----------------------------------------------------------------------------
/// The distance from each non-zero pixel of \p mask to the nearest zero.
///
/// `cv::distanceTransform` with `DIST_L2` and a mask size of 3. Zero pixels
/// come back zero; everything else is its chamfer distance out of the
/// background.
template < typename T >
viame::image_of< float >
distance_transform( viame::image_of< T > const& mask )
{
  if( mask.depth() != 1 )
  {
    throw std::invalid_argument( "distance_transform takes a single plane" );
  }

  auto const width = mask.width();
  auto const height = mask.height();

  viame::image_of< float > out( width, height, 1 );

  if( width == 0 || height == 0 )
  {
    return out;
  }

  // Outside the image is **not** background: a shape running off the edge is
  // not made close to anything by the edge, which is what OpenCV does and is
  // the right answer -- the pixels beyond are unknown, not empty.
  //
  // The consequence is that a mask with no zero anywhere has no answer, and
  // the saturated value is what comes back. `FLT_MAX` rather than something
  // smaller so that it is the same saturated value OpenCV returns; adding a
  // step cost to it rounds back to itself in float, so it survives both
  // passes without becoming an infinity.
  auto const far = std::numeric_limits< float >::max();

  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      out( x, y, 0 ) = ( mask( x, y, 0 ) != T{} ) ? far : 0.0f;
    }
  }

  auto const relax = [ & ]( float& best, size_t x, size_t y, float step )
  {
    best = std::min( best, out( x, y, 0 ) + step );
  };

  // Forward: north west, north, north east and west are already final
  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      auto best = out( x, y, 0 );

      if( best == 0.0f )
      {
        continue;
      }

      if( y > 0 )
      {
        if( x > 0 )         { relax( best, x - 1, y - 1, chamfer_diagonal ); }
        relax( best, x, y - 1, chamfer_orthogonal );
        if( x + 1 < width ) { relax( best, x + 1, y - 1, chamfer_diagonal ); }
      }

      if( x > 0 ) { relax( best, x - 1, y, chamfer_orthogonal ); }

      out( x, y, 0 ) = best;
    }
  }

  // Backward: the other four neighbours, over the forward pass's answers
  for( size_t y = height; y-- > 0; )
  {
    for( size_t x = width; x-- > 0; )
    {
      auto best = out( x, y, 0 );

      if( best == 0.0f )
      {
        continue;
      }

      if( y + 1 < height )
      {
        if( x > 0 )         { relax( best, x - 1, y + 1, chamfer_diagonal ); }
        relax( best, x, y + 1, chamfer_orthogonal );
        if( x + 1 < width ) { relax( best, x + 1, y + 1, chamfer_diagonal ); }
      }

      if( x + 1 < width ) { relax( best, x + 1, y, chamfer_orthogonal ); }

      out( x, y, 0 ) = best;
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_DISTANCE_H
