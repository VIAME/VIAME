/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_HOMOGRAPHY_OVERLAP_H
#define VIAME_IMAGE_OPS_HOMOGRAPHY_OVERLAP_H

#include <image_ops/polygon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Where two segments cross, if they do at a single point.
///
/// Parallel and collinear segments report no crossing: a collinear overlap
/// has no single point to report, and its endpoints are already picked up as
/// corners inside the other quad.
inline bool
segment_intersection( std::pair< double, double > const& a0,
                      std::pair< double, double > const& a1,
                      std::pair< double, double > const& b0,
                      std::pair< double, double > const& b1,
                      std::pair< double, double >& out )
{
  auto const ax = a1.first - a0.first;
  auto const ay = a1.second - a0.second;
  auto const bx = b1.first - b0.first;
  auto const by = b1.second - b0.second;

  auto const denominator = ax * by - ay * bx;

  if( std::abs( denominator ) < 1e-12 )
  {
    return false;
  }

  auto const dx = b0.first - a0.first;
  auto const dy = b0.second - a0.second;

  auto const t = ( dx * by - dy * bx ) / denominator;
  auto const u = ( dx * ay - dy * ax ) / denominator;

  if( t < 0.0 || t > 1.0 || u < 0.0 || u > 1.0 )
  {
    return false;
  }

  out = { a0.first + t * ax, a0.second + t * ay };
  return true;
}

// ----------------------------------------------------------------------------
/// The convex hull of \p points, counter-clockwise, by monotone chain.
inline polygon
convex_hull( polygon points )
{
  if( points.size() < 3 )
  {
    return points;
  }

  std::sort( points.begin(), points.end() );
  points.erase( std::unique( points.begin(), points.end() ), points.end() );

  if( points.size() < 3 )
  {
    return points;
  }

  auto const cross =
    []( std::pair< double, double > const& o,
        std::pair< double, double > const& a,
        std::pair< double, double > const& b )
    {
      return ( a.first - o.first ) * ( b.second - o.second ) -
             ( a.second - o.second ) * ( b.first - o.first );
    };

  polygon hull( 2 * points.size() );
  size_t count = 0;

  for( auto const& point : points )
  {
    while( count >= 2 &&
           cross( hull[ count - 2 ], hull[ count - 1 ], point ) <= 0 )
    {
      --count;
    }
    hull[ count++ ] = point;
  }

  auto const lower = count + 1;

  for( size_t index = points.size() - 1; index > 0; --index )
  {
    auto const& point = points[ index - 1 ];

    while( count >= lower &&
           cross( hull[ count - 2 ], hull[ count - 1 ], point ) <= 0 )
    {
      --count;
    }
    hull[ count++ ] = point;
  }

  hull.resize( count - 1 );
  return hull;
}

// ----------------------------------------------------------------------------
/// Area of a simple polygon, by the shoelace formula.
inline double
polygon_area( polygon const& points )
{
  if( points.size() < 3 )
  {
    return 0.0;
  }

  double total = 0.0;

  for( size_t index = 0; index < points.size(); ++index )
  {
    auto const& a = points[ index ];
    auto const& b = points[ ( index + 1 ) % points.size() ];
    total += a.first * b.second - b.first * a.second;
  }

  return std::abs( total ) / 2.0;
}

// ----------------------------------------------------------------------------
/// Fraction of an \p ni by \p nj frame still covered after the homography.
///
/// The frame's corners are warped, and the overlap is the area shared by the
/// warped quad and the original one, over the frame area. The shared region
/// is built from the corners of each quad that fall inside the other plus the
/// crossings of their edges, and its area is taken from the convex hull of
/// those points. Both quads are convex whenever the homography is, so the
/// hull is the intersection rather than an over-estimate of it.
///
/// \param h  Row major 3x3 homography.
/// \return   0 for a degenerate or disjoint result, 1 for the identity.
inline double
homography_overlap( double const h[ 9 ], unsigned ni, unsigned nj )
{
  if( ni == 0 || nj == 0 )
  {
    return 0.0;
  }

  bool identity = true;
  for( size_t row = 0; row < 3 && identity; ++row )
  {
    for( size_t column = 0; column < 3; ++column )
    {
      auto const expected = ( row == column ) ? 1.0 : 0.0;

      if( h[ row * 3 + column ] != expected )
      {
        identity = false;
        break;
      }
    }
  }

  if( identity )
  {
    return 1.0;
  }

  polygon const frame = {
    { 0.0, 0.0 },
    { static_cast< double >( ni ), 0.0 },
    { static_cast< double >( ni ), static_cast< double >( nj ) },
    { 0.0, static_cast< double >( nj ) },
  };

  polygon warped( 4 );

  for( size_t index = 0; index < 4; ++index )
  {
    auto const x = frame[ index ].first;
    auto const y = frame[ index ].second;

    auto const wx = h[ 0 ] * x + h[ 1 ] * y + h[ 2 ];
    auto const wy = h[ 3 ] * x + h[ 4 ] * y + h[ 5 ];
    auto const ww = h[ 6 ] * x + h[ 7 ] * y + h[ 8 ];

    if( ww == 0.0 )
    {
      return 0.0;
    }

    warped[ index ] = { wx / ww, wy / ww };
  }

  polygon shared;

  for( size_t index = 0; index < 4; ++index )
  {
    if( contains( frame, warped[ index ].first, warped[ index ].second ) )
    {
      shared.push_back( warped[ index ] );
    }

    if( contains( warped, frame[ index ].first, frame[ index ].second ) )
    {
      shared.push_back( frame[ index ] );
    }
  }

  for( size_t i = 0; i < 4; ++i )
  {
    for( size_t j = 0; j < 4; ++j )
    {
      std::pair< double, double > crossing;

      if( segment_intersection( warped[ i ], warped[ ( i + 1 ) % 4 ],
                                frame[ j ], frame[ ( j + 1 ) % 4 ],
                                crossing ) )
      {
        shared.push_back( crossing );
      }
    }
  }

  if( shared.size() < 3 )
  {
    return 0.0;
  }

  auto const hull = convex_hull( shared );

  if( hull.size() < 3 )
  {
    return 0.0;
  }

  auto const frame_area = static_cast< double >( ni ) * nj;

  return frame_area > 0.0 ? polygon_area( hull ) / frame_area : 0.0;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_HOMOGRAPHY_OVERLAP_H
