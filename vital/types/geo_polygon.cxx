// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a geo polygon.
 */

#include "geo_polygon.h"
#include "geodesy.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace kwiver {
namespace vital {

using geo_raw_polygon_t = geo_polygon::geo_raw_polygon_t;

// ----------------------------------------------------------------------------
geo_polygon::
geo_polygon()
  : m_original_crs{ -1 }
{ }

// ----------------------------------------------------------------------------
geo_polygon::
geo_polygon( geo_raw_polygon_t const& polygon, int crs )
  : m_original_crs( crs )
{
  m_poly.insert( std::make_pair( crs, polygon ) );
}

// ----------------------------------------------------------------------------
bool geo_polygon
::is_empty() const
{
  return m_poly.empty();
}

// ----------------------------------------------------------------------------
geo_raw_polygon_t geo_polygon
::polygon() const
{
  return m_poly.at( m_original_crs );
}

// ----------------------------------------------------------------------------
int geo_polygon
::crs() const
{
  return m_original_crs;
}

// ----------------------------------------------------------------------------
geo_raw_polygon_t geo_polygon
::polygon( int crs ) const
{
  auto const i = m_poly.find( crs );
  if ( i == m_poly.end() )
  {
    auto new_poly = geo_raw_polygon_t{};
    auto const verts = polygon().get_vertices();

    for ( auto& v : verts )
    {
      new_poly.push_back( geo_conv( v, m_original_crs, crs ) );
    }
    m_poly.emplace( crs, new_poly );
    return new_poly;
  }

  return i->second;
}

// ----------------------------------------------------------------------------
void geo_polygon
::set_polygon( geo_raw_polygon_t const& poly, int crs )
{
  m_original_crs = crs;
  m_poly.clear();
  m_poly.insert( std::make_pair( crs, poly ) );
}

// ----------------------------------------------------------------------------
template<>
geo_polygon
config_block_get_value_cast( config_block_value_t const& value )
{
  // Remove trailing spaces so we can reliably test when we run out of content
  auto i = std::find_if( value.rbegin(), value.rend(),
                         []( char c ) { return !std::isspace( c ); } );
  std::stringstream s( std::string{ value.begin(), i.base() } );

  // Check for empty value
  if ( s.peek(), s.eof() ) {
    return {};
  }

  // Set up helper lambda to check for errors
  auto try_or_die = [&value]( std::istream& in ) {
    if ( in.fail() ) {
      VITAL_THROW( bad_config_block_cast,
                   "failed to convert from string representation \"" + value + "\"" );
    }
  };

  int crs;
  try_or_die( s >> crs );

  // Get points
  geo_raw_polygon_t verts;
  do
  {
    double x, y;
    try_or_die( s >> x );
    try_or_die( s >> y );
    verts.push_back( { x, y } );
  } while ( !s.eof() );

  // Return geodetic polygon
  return { verts, crs };
}

// ----------------------------------------------------------------------------
template<>
config_block_value_t
config_block_set_value_cast( geo_polygon const& value )
{
  // Handle empty polygon
  if ( value.is_empty() )
  {
    return {};
  }

  // Write CRS
  std::stringstream str_result;
  str_result << value.crs();

  // Determine appropriate output precision
  auto const& verts = value.polygon().get_vertices();
  auto const magnitude = std::accumulate(
    verts.begin(), verts.end(), 0.0,
    []( double cur, vector_2d const& p ) {
      return std::max( { cur, std::fabs( p[0] ), std::fabs( p[1] ) } );
    });
  auto const integer_digits =
    static_cast<int>( std::floor( std::log10( magnitude ) ) ) + 1;
  auto const total_digits =
    std::numeric_limits<double>::digits10 + 2;

  str_result.precision( std::max( 0, total_digits - integer_digits ) );
  str_result.setf( std::ios::fixed );

  // Write vertex coordinates
  for ( auto const& v : verts )
  {
    str_result << " " << v[0] << " " << v[1];
  }

  // Return result
  return str_result.str();
}

// ----------------------------------------------------------------------------
std::ostream&
operator<<( std::ostream& str, vital::geo_polygon const& obj )
{
  if ( obj.is_empty() )
  {
    str << "{ empty }";
  }
  else
  {
    auto const old_prec = str.precision();
    auto const verts = obj.polygon();

    str.precision( std::numeric_limits<double>::digits10 + 2 );
    str << "{";
    for ( size_t n = 0; n < verts.num_vertices(); ++n )
    {
      if ( n ) {
        str << ",";
      }
      auto const& v = verts.at( n );
      str << " " << v[0] << " / " << v[1];
    }
    str << " } @ " << obj.crs();

    str.precision( old_prec );
  }

  return str;
}

} } // end namespace
