/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
  if( m_original_crs >= 0 )
  {
    return m_poly.at( m_original_crs );
  }

  return geo_raw_polygon_t();
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
  auto try_or_die = [&value]( std::istream& s ) {
    if ( s.fail() ) {
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
