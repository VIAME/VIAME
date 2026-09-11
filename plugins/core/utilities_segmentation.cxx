/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "utilities_segmentation.h"

#include <image_ops/contours.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace viame {

namespace {

// Template implementation of simplify_polygon algorithm
template< typename PointType, typename CoordType >
std::vector< PointType >
simplify_polygon_impl( std::vector< PointType > const& curve, size_t max_points )
{
  // Modified Ramer-Douglas-Peucker. Instead of keeping points out of
  // tolerance, we add points until we reach the max.
  size_t size = curve.size();
  max_points = std::max( max_points, size_t( 2 ) );
  if( size <= max_points )
  {
    return curve;
  }

  // Find approximate diameter endpoints
  size_t start = 0, opposite;
  for( int diameter_iter = 0; diameter_iter < 3; ++diameter_iter )
  {
    auto const& ps = curve[ start ];
    size_t i_max = start;
    CoordType sq_dist_max = 0;
    for( size_t i = 0; i < size; ++i )
    {
      CoordType dx = curve[ i ][ 0 ] - ps[ 0 ];
      CoordType dy = curve[ i ][ 1 ] - ps[ 1 ];
      CoordType sq_dist = dx * dx + dy * dy;
      if( sq_dist > sq_dist_max )
      {
        i_max = i;
        sq_dist_max = sq_dist;
      }
    }
    opposite = start;
    start = i_max;
  }

  // Indices for rec and find_max are relative to start
  auto to_rel = [&]( size_t i ) { return ( i + size - start ) % size; };
  auto from_rel = [&]( size_t i ) { return ( i + start ) % size; };

  struct rec
  {
    double sq_dist;
    size_t l, r, i;
    bool operator <( rec const& other ) const
    {
      return this->sq_dist < other.sq_dist;
    }
  };

  auto find_max = [&]( size_t l, size_t r )
  {
    auto const& pl = curve[ from_rel( l ) ];
    auto const& pr = curve[ from_rel( r ) ];
    double dx = static_cast< double >( pr[ 0 ] ) - static_cast< double >( pl[ 0 ] );
    double dy = static_cast< double >( pr[ 1 ] ) - static_cast< double >( pl[ 1 ] );
    double sq_dist_den = dx * dx + dy * dy;

    auto sqrt_sq_dist_num = [&]( size_t i )
    {
      auto const& p = curve[ from_rel( i ) ];
      double px = static_cast< double >( p[ 0 ] ) - static_cast< double >( pl[ 0 ] );
      double py = static_cast< double >( p[ 1 ] ) - static_cast< double >( pl[ 1 ] );
      return std::abs( px * dy - py * dx );
    };

    size_t i = l + 1;
    size_t i_max = i;
    double ssdn_max = sqrt_sq_dist_num( i );

    for( ++i; i < r; ++i )
    {
      auto ssdn = sqrt_sq_dist_num( i );
      if( ssdn > ssdn_max )
      {
        i_max = i;
        ssdn_max = ssdn;
      }
    }
    return rec{ ssdn_max * ssdn_max / sq_dist_den, l, r, i_max };
  };

  // Initialize using the two approximate diameter endpoints and the
  // parts of the curve between them.
  std::vector< bool > keep( size, false );
  keep[ start ] = keep[ opposite ] = true;
  std::priority_queue< rec > queue;
  queue.push( find_max( 0, to_rel( opposite ) ) );
  queue.push( find_max( to_rel( opposite ), size ) );

  for( size_t keep_count = 2; keep_count < max_points; ++keep_count )
  {
    auto max_rec = queue.top();
    queue.pop();
    keep[ from_rel( max_rec.i ) ] = true;
    if( max_rec.i - max_rec.l > 1 )
    {
      queue.push( find_max( max_rec.l, max_rec.i ) );
    }
    if( max_rec.r - max_rec.i > 1 )
    {
      queue.push( find_max( max_rec.i, max_rec.r ) );
    }
  }

  std::vector< PointType > result;
  for( size_t i = 0; i < size; ++i )
  {
    if( keep[ i ] )
    {
      result.push_back( curve[ i ] );
    }
  }
  return result;
}

} // anonymous namespace

// Integer point version
std::vector< kwiver::vital::point_2i >
simplify_polygon( std::vector< kwiver::vital::point_2i > const& curve,
                  size_t max_points )
{
  return simplify_polygon_impl< kwiver::vital::point_2i, int >( curve, max_points );
}

// Double point version
std::vector< kwiver::vital::point_2d >
simplify_polygon( std::vector< kwiver::vital::point_2d > const& curve,
                  size_t max_points )
{
  return simplify_polygon_impl< kwiver::vital::point_2d, double >( curve, max_points );
}

// ----------------------------------------------------------------------------
std::vector< mask_contour >
mask_to_contours( kwiver::vital::image const& mask,
                  double tolerance, int max_points )
{
  namespace io = viame::image_ops;

  std::vector< mask_contour > out;

  if( mask.width() == 0 || mask.height() == 0 )
  {
    return out;
  }

  kwiver::vital::image_of< uint8_t > const typed( mask );

  for( auto const& traced : io::find_borders( typed ) )
  {
    auto const contour = io::simplify_chain( traced.points );

    if( contour.empty() )
    {
      continue;
    }

    // The tolerance is relative to the shorter side of this contour's own
    // bounding box, which is what the writers computed.
    long x_min = contour[ 0 ].i, x_max = contour[ 0 ].i;
    long y_min = contour[ 0 ].j, y_max = contour[ 0 ].j;

    for( auto const& at : contour )
    {
      x_min = std::min( x_min, at.i );
      x_max = std::max( x_max, at.i );
      y_min = std::min( y_min, at.j );
      y_max = std::max( y_max, at.j );
    }

    std::vector< kwiver::vital::point_2i > simplified;

    if( tolerance >= 0 )
    {
      double const scale =
        static_cast< double >( std::min( x_max - x_min + 1,
                                         y_max - y_min + 1 ) );

      for( auto const& at : io::approx_poly( contour, tolerance * scale ) )
      {
        simplified.emplace_back( static_cast< int >( at.i ),
                                 static_cast< int >( at.j ) );
      }
    }
    else
    {
      std::vector< kwiver::vital::point_2i > points;
      points.reserve( contour.size() );

      for( auto const& at : contour )
      {
        points.emplace_back( static_cast< int >( at.i ),
                             static_cast< int >( at.j ) );
      }

      simplified = simplify_polygon(
        points, static_cast< size_t >( std::max( 0, max_points ) ) );
    }

    out.push_back( mask_contour{ std::move( simplified ), traced.is_hole } );
  }

  return out;
}

} // end namespace viame
