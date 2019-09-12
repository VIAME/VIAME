/*ckwg +29
 * Copyright 2017, 2019 by Kitware, Inc.
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
 * \brief PROJ geo_conversion functor implementation
 */

#include "geo_conv.h"

#include <proj_api.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace proj {

namespace {

// ----------------------------------------------------------------------------
std::unordered_map< std::string, std::string >
extract_props( std::string text )
{
  std::unordered_map< std::string, std::string > props;

  while ( text.size() )
  {
    auto const i = text.find( ' ' );
    auto tok = text.substr( 0, i );
    text = ( i == std::string::npos ? std::string{} : text.substr( i + 1 ) );

    if ( tok.size() && tok[0] == '+' )
    {
      auto const j = tok.find( '=' );
      if ( j == std::string::npos )
      {
        props.emplace( tok.substr( 1 ), std::string{} );
      }
      else
      {
        props.emplace( tok.substr( 1, j - 1 ), tok.substr( j + 1 ) );
      }
    }
  }

  return props;
}

}

// ----------------------------------------------------------------------------
geo_conversion
::~geo_conversion()
{
  for ( auto i : m_projections )
  {
    pj_free( i.second );
  }
}

// ----------------------------------------------------------------------------
char const* geo_conversion
::id() const
{
  return "proj";
}

// ----------------------------------------------------------------------------
vital::geo_crs_description_t geo_conversion
::describe( int crs )
{
  static const auto prop_map =
    std::unordered_map< std::string, std::string >{
      { "datum", "datum" },
      { "ellps", "ellipse" },
      { "proj", "projection" },
      { "units", "units" },
    };

  // Get CRS init string
  auto const proj = projection( crs );
  auto const text = std::string{ pj_get_def( proj, 0 ) };

  // Parse init string into property key/value pairs
  auto const props = extract_props( text );

  // Convert to human-readable result
  vital::geo_crs_description_t result;
  for ( auto const iter : props )
  {
    if ( iter.first == "zone" )
    {
      result.emplace( "zone", iter.second );
      result.emplace( "hemisphere",
                      props.count( "south" ) ? "south" : "north" );
    }
    else
    {
      auto const prop_map_iter = prop_map.find( iter.first );
      if ( prop_map_iter != prop_map.end() )
      {
        result.emplace( prop_map_iter->second, iter.second );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
vital::vector_2d geo_conversion
::operator()( vital::vector_2d const& point, int from, int to )
{
  auto const proj_from = projection( from );
  auto const proj_to = projection( to );

  auto x = point[0];
  auto y = point[1];
  auto z = 0.0;

  if ( pj_is_latlong( proj_from ) )
  {
    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;
  }

  int err = pj_transform( proj_from, proj_to, 1, 1, &x, &y, &z );
  if ( err )
  {
    auto const msg =
      std::string{ "PROJ conversion failed: error " } + std::to_string( err );
    throw std::runtime_error( msg );
  }

  if ( pj_is_latlong( proj_to ) )
  {
    x *= RAD_TO_DEG;
    y *= RAD_TO_DEG;
  }

  return { x, y };
}

// ----------------------------------------------------------------------------
vital::vector_3d geo_conversion
::operator()( vital::vector_3d const& point, int from, int to )
{
  auto const proj_from = projection( from );
  auto const proj_to = projection( to );

  auto x = point[0];
  auto y = point[1];
  auto z = point[2];

  if ( pj_is_latlong( proj_from ) )
  {
    x *= DEG_TO_RAD;
    y *= DEG_TO_RAD;
  }

  int err = pj_transform( proj_from, proj_to, 1, 1, &x, &y, &z );
  if ( err )
  {
    auto const msg =
      std::string{ "PROJ conversion failed: error " } + std::to_string( err );
    throw std::runtime_error( msg );
  }

  if ( pj_is_latlong( proj_to ) )
  {
    x *= RAD_TO_DEG;
    y *= RAD_TO_DEG;
  }

  return { x, y, z };
}

// ----------------------------------------------------------------------------
void* geo_conversion
::projection( int crs )
{
  auto const i = m_projections.find( crs );

  if ( i == m_projections.end() )
  {
    auto const crs_str = std::to_string( crs );
    auto const arg = std::string{ "+init=epsg:" } + crs_str;
    auto const p = pj_init_plus( arg.c_str() );

    if ( ! p )
    {
      auto const msg =
        "Failed to construct PROJ projection for EPSG:" + crs_str;
      throw std::runtime_error( msg );
    }

    m_projections.emplace( crs, p );
    return p;
  }

  return i->second;
}

} } } // end namespace
