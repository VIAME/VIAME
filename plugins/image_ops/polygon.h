/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_POLYGON_H
#define VIAME_IMAGE_OPS_POLYGON_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace viame {
namespace image_ops {

/// A closed polygon as its vertices, in order.
typedef std::vector< std::pair< double, double > > polygon;

// ----------------------------------------------------------------------------
/// Whether a point lies on the segment from \p a to \p b.
inline bool
on_segment( std::pair< double, double > const& a,
            std::pair< double, double > const& b,
            double x, double y )
{
  auto const cross = ( b.first - a.first ) * ( y - a.second ) -
                     ( b.second - a.second ) * ( x - a.first );

  // The scale of the cross product grows with the segment length, so the
  // tolerance has to as well or a long edge rejects points that are on it
  auto const length = std::max(
    std::abs( b.first - a.first ), std::abs( b.second - a.second ) );

  if( std::abs( cross ) > 1e-9 * std::max( 1.0, length ) )
  {
    return false;
  }

  return x >= std::min( a.first, b.first ) - 1e-9 &&
         x <= std::max( a.first, b.first ) + 1e-9 &&
         y >= std::min( a.second, b.second ) - 1e-9 &&
         y <= std::max( a.second, b.second ) + 1e-9;
}

// ----------------------------------------------------------------------------
/// Whether the point is inside the polygon or anywhere on its boundary.
///
/// Even-odd ray casting, with the boundary tested separately because a ray
/// that grazes an edge is ambiguous. Including the boundary is what makes a
/// polygon whose vertices sit at x = 0 and x = 4 cover five columns rather
/// than four, matching `vgl_polygon_scan_iterator` and so the annotation
/// masks the VIAME CSV readers have always produced.
inline bool
contains( polygon const& points, double x, double y )
{
  if( points.size() < 3 )
  {
    return false;
  }

  bool inside = false;

  for( size_t index = 0; index < points.size(); ++index )
  {
    auto const& a = points[ index ];
    auto const& b = points[ ( index + 1 ) % points.size() ];

    if( on_segment( a, b, x, y ) )
    {
      return true;
    }

    // Half open in y, so a vertex shared by two edges is crossed once
    if( ( a.second > y ) != ( b.second > y ) )
    {
      auto const t = ( y - a.second ) / ( b.second - a.second );

      if( x < a.first + t * ( b.first - a.first ) )
      {
        inside = !inside;
      }
    }
  }

  return inside;
}

// ----------------------------------------------------------------------------
/// Call \p emit( x, y ) for every pixel of a \p width by \p height grid the
/// polygon covers, boundary included.
///
/// Only the polygon's own bounding box is visited, so the cost follows the
/// annotation rather than the frame.
template < typename Emit >
void
rasterize_polygon( polygon const& points, size_t width, size_t height,
                   Emit&& emit )
{
  if( points.size() < 3 || width == 0 || height == 0 )
  {
    return;
  }

  auto min_x = points[ 0 ].first;
  auto max_x = points[ 0 ].first;
  auto min_y = points[ 0 ].second;
  auto max_y = points[ 0 ].second;

  for( auto const& point : points )
  {
    min_x = std::min( min_x, point.first );
    max_x = std::max( max_x, point.first );
    min_y = std::min( min_y, point.second );
    max_y = std::max( max_y, point.second );
  }

  auto const first_x = std::max( 0, static_cast< int >( std::floor( min_x ) ) );
  auto const first_y = std::max( 0, static_cast< int >( std::floor( min_y ) ) );
  auto const last_x =
    std::min( static_cast< int >( width ) - 1,
              static_cast< int >( std::ceil( max_x ) ) );
  auto const last_y =
    std::min( static_cast< int >( height ) - 1,
              static_cast< int >( std::ceil( max_y ) ) );

  for( int y = first_y; y <= last_y; ++y )
  {
    for( int x = first_x; x <= last_x; ++x )
    {
      if( contains( points, x, y ) )
      {
        emit( x, y );
      }
    }
  }
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_POLYGON_H
