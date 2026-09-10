/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Drawing on an image
///
/// What `cv::rectangle`, `cv::line`, `cv::circle`, `cv::fillPoly` and
/// `cv::putText` did: `draw_detected_object_set`, `ocv_write`'s chips and
/// the debug overlays.
///
/// Geometry only, no anti-aliasing. `cv::LINE_8` is what every VIAME caller
/// asks for -- the default -- and an anti-aliased overlay would be a change
/// in behaviour rather than an improvement, since the pixels underneath are
/// what a person is trying to see.
///
/// A colour here is one value per plane, so the same call works on a grey
/// mask and a colour frame; a single value is broadcast to every plane.

#ifndef VIAME_IMAGE_OPS_DRAW_H
#define VIAME_IMAGE_OPS_DRAW_H

#include <image_ops/contours.h>
#include <image_ops/font_5x7.h>
#include <image_ops/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// One value per plane, or one value for all of them.
using colour = std::vector< double >;

namespace detail {

/// The value \p paint gives to \p plane.
inline double
plane_value( colour const& paint, size_t plane )
{
  if( paint.empty() )
  {
    return 0.0;
  }

  return ( plane < paint.size() ) ? paint[ plane ] : paint.back();
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Set one pixel, if it is inside the image.
template < typename T >
void
draw_point( kwiver::vital::image_of< T >& image, long i, long j,
            colour const& paint )
{
  if( i < 0 || j < 0 || i >= static_cast< long >( image.width() ) ||
      j >= static_cast< long >( image.height() ) )
  {
    return;
  }

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    image( static_cast< size_t >( i ), static_cast< size_t >( j ), plane ) =
      saturate_pixel< T >( detail::plane_value( paint, plane ) );
  }
}

// ----------------------------------------------------------------------------
/// A line from (\p i0, \p j0) to (\p i1, \p j1), \p thickness pixels wide.
///
/// Bresenham, which is what `cv::LINE_8` is. Thickness is applied by drawing
/// a filled square of that size at each step -- the same approximation
/// OpenCV makes for a thin line, and the only one that matters at the one to
/// three pixels an overlay uses.
template < typename T >
void
draw_line( kwiver::vital::image_of< T >& image, long i0, long j0, long i1,
           long j1, colour const& paint, long thickness = 1 )
{
  thickness = std::max( 1L, thickness );
  auto const half = ( thickness - 1 ) / 2;

  auto stamp = [ & ]( long i, long j )
  {
    if( thickness == 1 )
    {
      draw_point( image, i, j, paint );
      return;
    }

    for( long dj = -half; dj < thickness - half; ++dj )
    {
      for( long di = -half; di < thickness - half; ++di )
      {
        draw_point( image, i + di, j + dj, paint );
      }
    }
  };

  auto const di = std::abs( i1 - i0 );
  auto const dj = -std::abs( j1 - j0 );
  auto const step_i = ( i0 < i1 ) ? 1L : -1L;
  auto const step_j = ( j0 < j1 ) ? 1L : -1L;

  auto error = di + dj;

  for( ;; )
  {
    stamp( i0, j0 );

    if( i0 == i1 && j0 == j1 )
    {
      break;
    }

    auto const twice = 2 * error;

    if( twice >= dj )
    {
      error += dj;
      i0 += step_i;
    }

    if( twice <= di )
    {
      error += di;
      j0 += step_j;
    }
  }
}

// ----------------------------------------------------------------------------
/// A rectangle, outlined at \p thickness or filled when \p thickness is
/// negative -- which is `cv::FILLED`, and what OpenCV means by a negative
/// thickness too.
///
/// The rectangle is half open, as `rect` is: \p bounds.right is one past the
/// last column drawn.
template < typename T >
void
draw_rect( kwiver::vital::image_of< T >& image, rect const& bounds,
           colour const& paint, long thickness = 1 )
{
  if( bounds.empty() )
  {
    return;
  }

  auto const left = bounds.left;
  auto const top = bounds.top;
  auto const right = bounds.right - 1;
  auto const bottom = bounds.bottom - 1;

  if( thickness < 0 )
  {
    for( long j = top; j <= bottom; ++j )
    {
      for( long i = left; i <= right; ++i )
      {
        draw_point( image, i, j, paint );
      }
    }

    return;
  }

  draw_line( image, left, top, right, top, paint, thickness );
  draw_line( image, right, top, right, bottom, paint, thickness );
  draw_line( image, right, bottom, left, bottom, paint, thickness );
  draw_line( image, left, bottom, left, top, paint, thickness );
}

// ----------------------------------------------------------------------------
/// A circle of \p radius about (\p centre_i, \p centre_j).
///
/// OpenCV's own rasterisation, error update included, because the obvious
/// midpoint circle draws a visibly different ring: OpenCV's steps `dy` every
/// iteration and decides `dx` from a running error, which leaves a one pixel
/// gap at each cardinal point where midpoint puts a five pixel flat. Forty
/// of the two hundred pixels of a radius nine ring differ between the two.
template < typename T >
void
draw_circle( kwiver::vital::image_of< T >& image, long centre_i,
             long centre_j, long radius, colour const& paint,
             long thickness = 1 )
{
  if( radius < 0 )
  {
    return;
  }

  auto stamp = [ & ]( long i, long j )
  {
    if( thickness <= 1 )
    {
      draw_point( image, i, j, paint );
      return;
    }

    auto const half = ( thickness - 1 ) / 2;

    for( long dj = -half; dj < thickness - half; ++dj )
    {
      for( long di = -half; di < thickness - half; ++di )
      {
        draw_point( image, i + di, j + dj, paint );
      }
    }
  };

  auto span = [ & ]( long from, long to, long j )
  {
    for( long i = from; i <= to; ++i )
    {
      draw_point( image, i, j, paint );
    }
  };

  long error = 0;
  long dx = radius;
  long dy = 0;
  long plus = 1;
  long minus = ( radius << 1 ) - 1;

  while( dx >= dy )
  {
    auto const j_near_low = centre_j - dy;
    auto const j_near_high = centre_j + dy;
    auto const j_far_low = centre_j - dx;
    auto const j_far_high = centre_j + dx;

    auto const i_far_low = centre_i - dx;
    auto const i_far_high = centre_i + dx;
    auto const i_near_low = centre_i - dy;
    auto const i_near_high = centre_i + dy;

    if( thickness < 0 )
    {
      span( i_far_low, i_far_high, j_near_low );
      span( i_far_low, i_far_high, j_near_high );
      span( i_near_low, i_near_high, j_far_low );
      span( i_near_low, i_near_high, j_far_high );
    }
    else
    {
      stamp( i_far_low, j_near_low );
      stamp( i_far_high, j_near_low );
      stamp( i_far_low, j_near_high );
      stamp( i_far_high, j_near_high );
      stamp( i_near_low, j_far_low );
      stamp( i_near_high, j_far_low );
      stamp( i_near_low, j_far_high );
      stamp( i_near_high, j_far_high );
    }

    ++dy;
    error += plus;
    plus += 2;

    // The branchless step OpenCV writes: `mask` is 0 when the error is at
    // or below zero and -1 otherwise, so `minus` is taken off only when the
    // error has grown past it
    auto const mask = ( error <= 0 ) ? 0L : -1L;

    error -= minus & mask;
    minus -= mask & 2;
    dx += mask;
  }
}

// ----------------------------------------------------------------------------
/// Fill the polygon through \p points, which is `cv::fillPoly`.
///
/// Even-odd scanline filling, and the polygon is closed for you.
///
/// The outline is drawn as well as the interior, which is what `cv::fillPoly`
/// does and is not what "fill" suggests: a scanline fill alone leaves the
/// boundary pixels out wherever an edge crosses a scanline between two pixel
/// centres, and on a four-sided polygon twenty by twenty that is twenty nine
/// pixels of a difference. Filling only the interior and calling it
/// `fillPoly` would be wrong in a way that looks right.
template < typename T >
void
fill_polygon( kwiver::vital::image_of< T >& image,
              std::vector< point > const& points, colour const& paint )
{
  if( points.size() < 3 )
  {
    return;
  }

  long top = points[ 0 ].j;
  long bottom = points[ 0 ].j;

  for( auto const& at : points )
  {
    top = std::min( top, at.j );
    bottom = std::max( bottom, at.j );
  }

  top = std::max( 0L, top );
  bottom = std::min( static_cast< long >( image.height() ) - 1, bottom );

  for( long j = top; j <= bottom; ++j )
  {
    std::vector< double > crossings;

    for( size_t at = 0; at < points.size(); ++at )
    {
      auto const& a = points[ at ];
      auto const& b = points[ ( at + 1 ) % points.size() ];

      // Half-open in j, so a vertex shared by two edges is counted once
      auto const lower = std::min( a.j, b.j );
      auto const upper = std::max( a.j, b.j );

      if( j < lower || j >= upper )
      {
        continue;
      }

      auto const t = static_cast< double >( j - a.j ) /
                     static_cast< double >( b.j - a.j );

      crossings.push_back( static_cast< double >( a.i ) +
                           t * static_cast< double >( b.i - a.i ) );
    }

    std::sort( crossings.begin(), crossings.end() );

    for( size_t pair = 0; pair + 1 < crossings.size(); pair += 2 )
    {
      auto const from =
        static_cast< long >( std::ceil( crossings[ pair ] ) );
      auto const to =
        static_cast< long >( std::floor( crossings[ pair + 1 ] ) );

      for( long i = from; i <= to; ++i )
      {
        draw_point( image, i, j, paint );
      }
    }
  }

  // And the boundary, which the scanline fill leaves out
  for( size_t at = 0; at < points.size(); ++at )
  {
    auto const& a = points[ at ];
    auto const& b = points[ ( at + 1 ) % points.size() ];

    draw_line( image, a.i, a.j, b.i, b.j, paint );
  }
}

// ----------------------------------------------------------------------------
/// The pixel size of \p text at \p scale, as `cv::getTextSize` reports it.
inline rect
text_size( std::string const& text, long scale = 1 )
{
  scale = std::max( 1L, scale );

  rect out;
  out.right = static_cast< long >( text.size() ) * font_advance * scale;
  out.bottom = font_height * scale;

  if( !text.empty() )
  {
    // The trailing blank column is not part of the last glyph
    out.right -= scale;
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Draw \p text with its **top left** at (\p i, \p j).
///
/// `cv::putText` places text by its baseline, which is the bottom left for
/// a glyph with no descender. Top left instead, because a bitmap font has no
/// baseline to speak of and every VIAME caller is placing a label against a
/// box edge; `text_size` gives the height to offset by for a caller that
/// wants the other convention.
///
/// The glyphs are `font_5x7.h`'s, not Hershey's, so text drawn here does not
/// look like text drawn by OpenCV. It is legible, which is what a debug
/// overlay needs.
template < typename T >
void
draw_text( kwiver::vital::image_of< T >& image, std::string const& text,
           long i, long j, colour const& paint, long scale = 1 )
{
  scale = std::max( 1L, scale );

  long pen = i;

  for( auto const ch : text )
  {
    for( int gj = 0; gj < font_height; ++gj )
    {
      for( int gi = 0; gi < font_width; ++gi )
      {
        if( !font_pixel( ch, gi, gj ) )
        {
          continue;
        }

        for( long sj = 0; sj < scale; ++sj )
        {
          for( long si = 0; si < scale; ++si )
          {
            draw_point( image, pen + gi * scale + si, j + gj * scale + sj,
                        paint );
          }
        }
      }
    }

    pen += font_advance * scale;
  }
}

} // namespace image_ops
} // namespace viame

#endif
