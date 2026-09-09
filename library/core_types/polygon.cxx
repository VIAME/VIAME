// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief core polygon implementation

#include "polygon.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace kwiver {

namespace vital {

namespace {

// ----------------------------------------------------------------------------
enum class polygon_combine_mode
{
  INTERSECTION,
  UNION,
};

// ----------------------------------------------------------------------------
std::vector< vital::vector_2d >
convex_combine(
  std::vector< vital::vector_2d > const& a_in,
  std::vector< vital::vector_2d > const& b_in,
  polygon_combine_mode combine_mode )
{
  // Determine maximum scalar value among all points
  auto max_value = 0.0;
  for( auto const& polygon : { a_in, b_in } )
  {
    for( auto const& p : polygon )
    {
      max_value = std::max(
        max_value,
        std::max( std::abs( p[ 0 ] ), std::abs( p[ 1 ] ) ) );
    }
  }

  // Consider two points "equal" if they are close enough for their difference
  // to probably be rounding error, relative to the overall scale of the inputs
  constexpr auto epsilon = 1.0e-15;
  auto const points_equal =
    [ max_value, epsilon ]( vital::vector_2d const& p0,
                            vital::vector_2d const& p1 ){
      return
        std::abs( p1[ 0 ] - p0[ 0 ] ) <= max_value * epsilon &&
        std::abs( p1[ 1 ] - p0[ 1 ] ) <= max_value * epsilon;
    };

  // Remove consecutive identical points and points that are exactly on a line
  // between the points on either side. This avoids various edge cases and does
  // not change the geometry of the polygon
  auto const remove_duplicates =
    [ points_equal, epsilon ]( std::vector< vital::vector_2d > const& points ){
      if( points.size() < 2 )
      {
        return points;
      }

      // Remove identical points
      std::vector< vital::vector_2d > pass1;
      for( size_t i = 0; i < points.size(); ++i )
      {
        auto const& p0 = points[ ( i + points.size() - 1 ) % points.size() ];
        auto const& p1 = points[ i ];
        if( !points_equal( p0, p1 ) )
        {
          pass1.emplace_back( p1 );
        }
      }

      if( pass1.size() <= 2 )
      {
        return pass1;
      }

      // Remove the middle point when three consecutive points are colinear
      std::vector< vital::vector_2d > pass2;
      for( size_t i = 0; i < pass1.size(); ++i )
      {
        auto const& p0 = pass1[ ( i + pass1.size() - 1 ) % pass1.size() ];
        auto const& p1 = pass1[ i ];
        auto const& p2 = pass1[ ( i + 1 ) % pass1.size() ];
        auto const t1 = ( p0[ 0 ] - p1[ 0 ] ) * ( p1[ 1 ] - p2[ 1 ] );
        auto const t2 = ( p0[ 1 ] - p1[ 1 ] ) * ( p1[ 0 ] - p2[ 0 ] );
        if( std::abs( t1 - t2 ) >
            std::max( std::abs( t1 ), std::abs( t2 ) ) * epsilon )
        {
          pass2.emplace_back( p1 );
        }
      }

      return pass2;
    };

  // Simplify the input polygons
  auto const a = remove_duplicates( a_in );
  auto const b = remove_duplicates( b_in );

  // Struct to hold pre-calculated information about an edge
  struct edge_info
  {
    edge_info( std::vector< vital::vector_2d > const& polygon, size_t index )
    {
      p0 = polygon[ index ];
      p1 = polygon[ ( index + 1 ) % polygon.size() ];
      v = p1 - p0;
      v_norm = v.normalized();
      perp_norm = { -v_norm[ 1 ], v_norm[ 0 ] };
      t0 = v_norm.dot( p0 );
      t1 = v_norm.dot( p1 );
      t_perp = perp_norm.dot( p0 );
      perp = perp_norm * t_perp;
    }

    // First point
    vital::vector_2d p0;
    // Second point
    vital::vector_2d p1;
    // Vector from first to second point
    vital::vector_2d v;
    // Normalized direction of the edge, from first to second point
    vital::vector_2d v_norm;
    // Shortest vector from (0, 0) to the infinite line that contains the edge.
    // Necessarily perpendicular to the edge.
    vital::vector_2d perp;
    // Normalized direction of the edge, rotated counter-clockwize 90 degrees.
    // Points "left", or "in" for a counter-clockwise polygon.
    vital::vector_2d perp_norm;
    // Scalar such that p0 == v_norm * t0 + perp
    double t0;
    // Scalar such that p1 == v_norm * t1 + perp
    double t1;
    // Scalar such that perp == perp_norm * t_perp
    double t_perp;
  };

  // Pre-calculate edge info
  std::vector< edge_info > a_edges;
  for( size_t i = 0; i < a.size(); ++i )
  {
    a_edges.emplace_back( a, i );
  }

  std::vector< edge_info > b_edges;
  for( size_t i = 0; i < b.size(); ++i )
  {
    b_edges.emplace_back( b, i );
  }

  // Return the intersection of two edges
  // Can return std::monostate (no intersection), a single point, or two points
  // (line segment)
  auto const edges_intersect =
    [ epsilon, &points_equal ](edge_info const& lhs, edge_info const& rhs) ->
    std::variant<
      std::monostate,
      vital::vector_2d,
      std::pair< vital::vector_2d, vital::vector_2d > > {
      // https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment
      auto const denominator1 =
        ( lhs.p0[ 0 ] - lhs.p1[ 0 ] ) *
        ( rhs.p0[ 1 ] - rhs.p1[ 1 ] );
      auto const denominator2 =
        ( lhs.p0[ 1 ] - lhs.p1[ 1 ] ) *
        ( rhs.p0[ 0 ] - rhs.p1[ 0 ] );
      auto const denominator = denominator1 - denominator2;

      if( std::abs( denominator ) <= std::max(
        std::abs( denominator1 ),
        std::abs( denominator2 ) ) * epsilon )
      {
        // Edges are parallel or colinear

        auto const lhs_t_perp = lhs.t_perp;
        auto const rhs_t_perp = lhs.perp_norm.dot( rhs.p0 );
        if( std::abs( lhs_t_perp - rhs_t_perp ) >
            std::max(
              std::abs( lhs_t_perp ),
              std::abs( rhs_t_perp ) ) * epsilon )
        {
          // Edges are parallel
          return std::monostate{};
        }

        auto rhs_t0 = lhs.v_norm.dot( rhs.p0 );
        auto rhs_t1 = lhs.v_norm.dot( rhs.p1 );
        if( rhs_t1 < rhs_t0 )
        {
          // Account for edges facing opposite directions
          std::swap( rhs_t0, rhs_t1 );
        }

        if( rhs_t0 > lhs.t1 || rhs_t1 < lhs.t0 )
        {
          // Edges are colinear but do not intersect
          return std::monostate{};
        }

        auto const t0 = std::max( lhs.t0, rhs_t0 );
        auto const t1 = std::min( lhs.t1, rhs_t1 );

        if( t1 - t0 <= std::max(
          std::abs( t0 ),
          std::abs( t1 ) ) * epsilon )
        {
          // Edges are colinear and intersect at a single point
          return lhs.perp + lhs.v_norm * t0;
        }

        // Edges are colinear and intersect for more than a single point
        return std::make_pair(
          lhs.perp + lhs.v_norm * t0,
          lhs.perp + lhs.v_norm * t1 );
      }

      // Edges' lines should intersect at a single point
      auto t =
        ( ( lhs.p0[ 0 ] - rhs.p0[ 0 ] ) * ( rhs.p0[ 1 ] - rhs.p1[ 1 ] ) -
          ( lhs.p0[ 1 ] - rhs.p0[ 1 ] ) * ( rhs.p0[ 0 ] - rhs.p1[ 0 ] ) ) /
        denominator;
      if( t < 0.0 )
      {
        if( points_equal( lhs.p0, lhs.p0 + lhs.v * t ) )
        {
          // Point of lines' intersection is lhs.p0, within rounding error
          t = 0.0;
        }
        else
        {
          // Point of lines' intersection not in bounds of lhs edge
          return std::monostate{};
        }
      }
      else if( t > 1.0 )
      {
        if( points_equal( lhs.p1, lhs.p0 + lhs.v * t ) )
        {
          // Point of lines' intersection is lhs.p1, within rounding error
          t = 1.0;
        }
        else
        {
          // Point of lines' intersection not in bounds of lhs edge
          return std::monostate{};
        }
      }

      auto u =
        ( ( lhs.p0[ 0 ] - lhs.p1[ 0 ] ) * ( lhs.p0[ 1 ] - rhs.p0[ 1 ] ) -
          ( lhs.p0[ 1 ] - lhs.p1[ 1 ] ) * ( lhs.p0[ 0 ] - rhs.p0[ 0 ] ) ) /
        -denominator;
      if( u < 0.0 )
      {
        if( points_equal( rhs.p0, rhs.p0 + rhs.v * u ) )
        {
          // Point of lines' intersection is rhs.p0, within rounding error
          u = 0.0;
        }
        else
        {
          // Point of lines' intersection not in bounds of second edge
          return std::monostate{};
        }
      }
      else if( u > 1.0 )
      {
        if( points_equal( rhs.p1, rhs.p0 + rhs.v * u ) )
        {
          // Point of lines' intersection is rhs.p1, within rounding error
          u = 1.0;
        }
        else
        {
          // Point of lines' intersection not in bounds of second edge
          return std::monostate{};
        }
      }

      // Point of lines' intersection in bounds of both edges
      return lhs.p0 + lhs.v * t;
    };

  // Return true if lhs points in the direction of rhs but they do not intersect
  auto const points_toward =
    [](edge_info const& lhs, edge_info const& rhs) -> bool {
      auto const forward_vector = rhs.v_norm * rhs.v_norm.dot( lhs.p1 );
      auto const projected = rhs.perp + forward_vector;
      auto const to_projected = projected - lhs.p1;
      return lhs.v_norm.dot( to_projected ) > 0;
    };

  // Return true if rhs is "to the left" of lhs
  auto const is_outside =
    [ epsilon ](edge_info const& lhs, edge_info const& rhs) -> bool {
      // Direction from lhs to rhs
      vital::vector_2d v = rhs.p1 - lhs.p1;
      if( v.isZero() )
      {
        v = rhs.p0 - lhs.p0;
      }

      std::array< double, 4 > values = {
        std::abs( v[ 0 ] ), std::abs( v[ 1 ] ),
        std::abs( lhs.perp_norm[ 0 ] ), std::abs( lhs.perp_norm[ 1 ] ) };

      auto const dot = lhs.perp_norm.dot( v );
      if( std::abs( dot ) >
          *std::max_element( values.begin(), values.end() ) * epsilon )
      {
        // Check if going from lhs to rhs is in the same direction as the lhs
        // edge's "leftward" direction
        return dot > 0;
      }
      else
      {
        // If our answer is within rounding error's distance of zero, use the
        // other edge's "leftward" direction just in case it gives a clearer
        // answer
        return rhs.perp_norm.dot( v ) > 0;
      }
    };

  // Return true if the point connecting consecutive edges edge1 and edge2,
  // which touches the one or two consecutive edge(s) in corner_edges,
  // constitutes the entire intersection between the two polygons
  auto const is_single_corner_intersection =
    [ &points_equal ]( edge_info const& edge1, edge_info const& edge2,
                       std::vector< edge_info const* > const& corner_edges ){
      bool edge1_outside = false;
      bool edge2_outside = false;
      for( auto const corner_edge : corner_edges )
      {
        edge1_outside |= corner_edge->perp_norm.dot( edge1.p0 ) <
                         corner_edge->t_perp;
        edge2_outside |= corner_edge->perp_norm.dot( edge2.p1 ) <
                         corner_edge->t_perp;
      }

      return edge1_outside && edge2_outside &&
             ( edge1.perp_norm.dot( corner_edges[ 0 ]->p0 ) < edge1.t_perp ||
               edge2.perp_norm.dot( corner_edges[ 0 ]->p0 ) < edge2.t_perp ) &&
             ( corner_edges.size() < 2 ||
               edge1.perp_norm.dot( corner_edges[ 1 ]->p1 ) < edge1.t_perp ||
               edge2.perp_norm.dot( corner_edges[ 1 ]->p1 ) < edge2.t_perp );
    };

  // Now that we've defined utility functions, set up initial state and start
  // the main loop

  std::vector< vital::vector_2d > result;
  size_t a_index = 0;
  size_t b_index = 0;
  size_t a_first = 0;
  size_t b_first = 0;
  bool a_looped = false;
  bool b_looped = false;
  bool follow_a = true;
  bool found_intersection = false;

  for( size_t i = 0; true; ++i )
  {
    if( i > ( a.size() + b.size() ) * 2 )
    {
      // This should never happen, but is put in as a failsafe to prevent an
      // infinite loop in case of some bug
      throw std::logic_error( "Failed to combine polygons" );
    }

    auto const edge_a = a_edges[ a_index ];
    auto const edge_b = b_edges[ b_index ];

    // Set up a few utility functions for the loop

    auto const increment_a =
      [ & ](){
        if( found_intersection && follow_a )
        {
          result.emplace_back( edge_a.p1 );
        }
        a_index = ( a_index + 1 ) % a.size();
        if( ( !found_intersection && a_index == 0 ) ||
            ( found_intersection && a_index == a_first ) )
        {
          a_looped = true;
        }
      };

    auto const increment_b =
      [ & ](){
        if( found_intersection && !follow_a )
        {
          result.emplace_back( edge_b.p1 );
        }
        b_index = ( b_index + 1 ) % b.size();
        if( ( !found_intersection && b_index == 0 ) ||
            ( found_intersection && b_index == b_first ) )
        {
          b_looped = true;
        }
      };

    auto const increment_outside =
      [ & ](){
        if( is_outside( edge_a, edge_b ) )
        {
          increment_a();
        }
        else
        {
          increment_b();
        }
      };

    auto const intersection = edges_intersect( edge_a, edge_b );

    if( !found_intersection &&
        !std::holds_alternative< std::monostate >( intersection ) )
    {
      // This is the first intersection we've found
      found_intersection = true;
      a_first = a_index;
      b_first = b_index;
    }

    if( std::holds_alternative< std::monostate >( intersection ) )
    {
      // No intersection; handling according to
      // https://tildesites.bowdoin.edu/~ltoma/teaching/cs3250-CompGeom/spring17/Lectures/cg-convexintersection.pdf

      if( points_toward( edge_a, edge_b ) )
      {
        if( points_toward( edge_b, edge_a ) )
        {
          increment_outside();
        }
        else
        {
          increment_a();
        }
      }
      else if( points_toward( edge_b, edge_a ) )
      {
        increment_b();
      }
      else
      {
        increment_outside();
      }
    }
    else if( auto const p = std::get_if< vital::vector_2d >( &intersection ) )
    {
      // Single-point intersection

      result.emplace_back( *p );

      if( points_equal( edge_a.p1, *p ) )
      {
        // Assemble edge(s) in B we intersect with
        std::vector< edge_info const* > corner_edges{ &edge_b };
        if( points_equal( edge_b.p0, *p ) )
        {
          corner_edges.insert(
            corner_edges.begin(),
            &b_edges[ ( b_index + b.size() - 1 ) % b.size() ] );
        }
        else if( points_equal( edge_b.p1, *p ) )
        {
          corner_edges.emplace_back( &b_edges[ ( b_index + 1 ) % b.size() ] );
        }

        auto const& next_a = a_edges[ ( a_index + 1 ) % a.size() ];
        if( is_single_corner_intersection( edge_a, next_a, corner_edges ) )
        {
          // This single point is the entire intersection between the two
          // polygons
          if( ( combine_mode == polygon_combine_mode::INTERSECTION ) )
          {
            // Return that point
            return { edge_a.p1 };
          }
          else
          {
            // Return the two polygons joined at this point
            result.clear();
            for( size_t j = 0; j <= a.size(); ++j )
            {
              result.emplace_back( a[ ( a_index + j + 1 ) % a.size() ] );
            }
            for( size_t j = 0; j < b.size(); ++j )
            {
              result.emplace_back( b[ ( b_index + j + 1 ) % b.size() ] );
            }
            return remove_duplicates( result );
          }
        }
        else if( points_equal( edge_b.p1, *p ) )
        {
          // Both edges intersect at their second point; check which of their
          // next edges is outside the other, in case that switches through this
          // point
          if( is_outside(
            a_edges[ ( a_index + 1 ) % a.size() ],
            b_edges[ ( b_index + 1 ) % b.size() ] ) )
          {
            follow_a = ( combine_mode != polygon_combine_mode::INTERSECTION );
            increment_a();
          }
          else
          {
            follow_a = ( combine_mode == polygon_combine_mode::INTERSECTION );
            increment_b();
          }
        }
        else
        {
          // No special case
          follow_a = ( combine_mode == polygon_combine_mode::INTERSECTION );
          increment_a();
        }
      }
      else if( points_equal( edge_b.p1, *p ) )
      {
        // Assemble edge(s) in A we intersect with
        std::vector< edge_info const* > corner_edges{ &edge_a };
        if( points_equal( edge_a.p0, *p ) )
        {
          corner_edges.insert(
            corner_edges.begin(),
            &a_edges[ ( a_index + a.size() - 1 ) % a.size() ] );
        }

        auto const& next_b = b_edges[ ( b_index + 1 ) % b.size() ];
        if( is_single_corner_intersection( edge_b, next_b, corner_edges ) )
        {
          // This single point is the entire intersection between the two
          // polygons
          if( ( combine_mode == polygon_combine_mode::INTERSECTION ) )
          {
            // Return that point
            return { edge_b.p1 };
          }
          else
          {
            // Return the two polygons joined at this point
            result.clear();
            for( size_t j = 0; j <= b.size(); ++j )
            {
              result.emplace_back( b[ ( b_index + j + 1 ) % b.size() ] );
            }
            for( size_t j = 0; j < a.size(); ++j )
            {
              result.emplace_back( a[ ( a_index + j + 1 ) % a.size() ] );
            }
            return remove_duplicates( result );
          }
        }
        else
        {
          // No special case
          follow_a = ( combine_mode != polygon_combine_mode::INTERSECTION );
          increment_b();
        }
      }
      // Edges intersect not through their end points; increment the outside
      else if( is_outside( edge_a, edge_b ) )
      {
        follow_a = ( combine_mode != polygon_combine_mode::INTERSECTION );
        increment_a();
      }
      else
      {
        follow_a = ( combine_mode == polygon_combine_mode::INTERSECTION );
        increment_b();
      }
    }
    else if( edge_a.v_norm.dot( edge_b.v_norm ) > 0 )
    {
      // Colinear intersection with edges facing same way

      if( edge_a.v_norm.dot( edge_a.p1 ) > edge_a.v_norm.dot( edge_b.p1 ) )
      {
        // A is further ahead
        follow_a = ( combine_mode != polygon_combine_mode::INTERSECTION );
        increment_b();
      }
      else
      {
        // B is further ahead, or they are equal
        follow_a = ( combine_mode == polygon_combine_mode::INTERSECTION );
        increment_a();
      }
    }
    else
    {
      // Colinear intersection with edges facing opposite ways

      auto const& [ p0, p1 ] =
        std::get< std::pair< vital::vector_2d, vital::vector_2d > >(
          intersection );
      if( combine_mode == polygon_combine_mode::INTERSECTION )
      {
        // This is necessarily the only intersection due to convexity and
        // remove_duplicates()
        return { p0, p1 };
      }
      // For union, we need to switch which polygon we are following
      else if( follow_a )
      {
        result.emplace_back( p0 );
        follow_a = false;
        increment_b();
      }
      else
      {
        result.emplace_back( p1 );
        follow_a = true;
        increment_a();
      }
    }

    // Check if we're done
    if( ( found_intersection && a_looped && b_looped ) ||
        ( !found_intersection && ( a_looped || b_looped ) ) )
    {
      break;
    }
  }

  // Last case: one polygon is entirely within the other with no edges touching
  if( !found_intersection )
  {
    // Check if B is entirely within A
    if( std::all_of(
      a_edges.begin(), a_edges.end(),
      [ &b ]( edge_info const& edge ){
        return edge.perp_norm.dot( b[ 0 ] - edge.p0 ) >= 0;
      } ) )
    {
      return ( combine_mode == polygon_combine_mode::INTERSECTION ) ? b : a;
    }

    // Check if A is entirely within B
    if( std::all_of(
      b_edges.begin(), b_edges.end(),
      [ &a ]( edge_info const& edge ){
        return edge.perp_norm.dot( a[ 0 ] - edge.p0 ) >= 0;
      } ) )
    {
      return ( combine_mode == polygon_combine_mode::INTERSECTION ) ? a : b;
    }
  }

  return remove_duplicates( result );
}

}  // namespace <anonymous>

// ----------------------------------------------------------------------------
polygon
::polygon()
{}

polygon
::polygon( const std::vector< point_t >& dat )
  : m_polygon( dat )
{}

polygon
::polygon( std::initializer_list< point_t > dat )
  : m_polygon( dat )
{}

// ----------------------------------------------------------------------------
polygon::
~polygon()
{}

// ----------------------------------------------------------------------------
void
polygon
::push_back( double x, double y )
{
  m_polygon.push_back( point_t( x, y ) );
}

// ----------------------------------------------------------------------------
void
polygon
::push_back( const kwiver::vital::polygon::point_t& pt )
{
  m_polygon.push_back( pt );
}

// ----------------------------------------------------------------------------
size_t
polygon
::num_vertices() const
{
  return m_polygon.size();
}

// ----------------------------------------------------------------------------
bool
polygon
::contains( double x, double y )
{
  bool c = false;

  int n = static_cast< int >( m_polygon.size() );
  for( int i = 0, j = n - 1; i < n; j = i++ )
  {
    const point_t& p_i = m_polygon[ i ];
    const point_t& p_j = m_polygon[ j ];

    // by definition, corner points and edge points are inside the polygon:
    if( ( p_j.x() - x ) * ( p_i.y() - y ) ==
        ( p_i.x() - x ) * ( p_j.y() - y ) &&
        ( ( ( p_i.x() <= x ) && ( x <= p_j.x() ) ) ||
          ( ( p_j.x() <= x ) && ( x <= p_i.x() ) ) ) &&
        ( ( ( p_i.y() <= y ) && ( y <= p_j.y() ) ) ||
          ( ( p_j.y() <= y ) && ( y <= p_i.y() ) ) ) )
    {
      return true;
    }

    // invert c for each edge crossing:
    if( ( ( ( p_i.y() <= y ) && ( y < p_j.y() ) ) ||
          ( ( p_j.y() <= y ) && ( y < p_i.y() ) ) ) &&
        ( x <
          ( p_j.x() - p_i.x() ) * ( y - p_i.y() ) / ( p_j.y() - p_i.y() ) +
          p_i.x() ) )
    {
      c = !c;
    }
  } // end for

  return c;
}

// ----------------------------------------------------------------------------
bool
polygon
::contains( const kwiver::vital::polygon::point_t& pt )
{
  return contains( pt[ 0 ], pt[ 1 ] );
}

// ----------------------------------------------------------------------------
kwiver::vital::polygon::point_t
polygon
::at( size_t idx ) const
{
  if( idx >= m_polygon.size() )
  {
    std::stringstream str;
    str << "Requested index " << idx
        << " is beyond the end of the polygon. Last valid index is "
        << m_polygon.size() - 1;
    throw std::out_of_range( str.str() );
  }

  return m_polygon[ idx ];
}

// ----------------------------------------------------------------------------
std::vector< kwiver::vital::polygon::point_t >
polygon
::get_vertices() const
{
  return m_polygon;
}

// ----------------------------------------------------------------------------
double
polygon
::area() const
{
  // Degenerate polygons
  if( m_polygon.size() < 3 )
  {
    return 0.0;
  }

  // Subtract origin to save precision
  auto const origin_y = m_polygon[ 0 ][ 1 ];

  // https://en.wikipedia.org/wiki/Shoelace_formula
  auto value = 0.0;
  for( size_t i = 0; i < m_polygon.size(); ++i )
  {
    auto const j = ( i + 1 ) % m_polygon.size();
    auto const& p0 = m_polygon[ i ];
    auto const& p1 = m_polygon[ j ];
    value +=
      ( ( p0[ 1 ] - origin_y ) + ( p1[ 1 ] - origin_y ) ) *
      ( p0[ 0 ] - p1[ 0 ] );
  }

  return 0.5 * value;
}

// ----------------------------------------------------------------------------
std::optional< polygon >
polygon
::convex_intersection( polygon const& a, polygon const& b )
{
  auto const result =
    convex_combine(
      a.get_vertices(), b.get_vertices(), polygon_combine_mode::INTERSECTION );

  if( result.empty() )
  {
    return std::nullopt;
  }

  return result;
}

// ----------------------------------------------------------------------------
std::optional< polygon >
polygon
::convex_union( polygon const& a, polygon const& b )
{
  auto const result =
    convex_combine(
      a.get_vertices(), b.get_vertices(), polygon_combine_mode::UNION );

  if( result.empty() )
  {
    return std::nullopt;
  }

  return result;
}

// ----------------------------------------------------------------------------
bool
operator==( polygon const& lhs, polygon const& rhs )
{
  return lhs.get_vertices() == rhs.get_vertices();
}

// ----------------------------------------------------------------------------
bool
operator!=( polygon const& lhs, polygon const& rhs )
{
  return !( lhs == rhs );
}

} // namespace vital

}      // end namespace
