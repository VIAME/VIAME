// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation of a geo point.
 */

#include "geodesy.h"

#include <atomic>
#include <cmath>

namespace kwiver {
namespace vital {

namespace {

static std::atomic< geo_conversion* > s_geo_conv;

// ----------------------------------------------------------------------------
double fmod( double n, double d )
{
  // Return the actual modulo `n % d`; std::fmod does the wrong thing for
  // negative numbers (rounds to zero, rather than rounding down)
  return n - ( d * std::floor( n / d ) );
}

} // end namespace

// ----------------------------------------------------------------------------
geo_conversion*
get_geo_conv()
{
  return s_geo_conv.load();
}

// ----------------------------------------------------------------------------
void
set_geo_conv( geo_conversion* c )
{
  s_geo_conv.store( c );
}

// ----------------------------------------------------------------------------
geo_crs_description_t
geo_crs_description( int crs )
{
  auto const c = s_geo_conv.load();
  if ( !c )
  {
    throw std::runtime_error( "No geo-conversion functor is registered" );
  }

  return c->describe( crs );
}

// ----------------------------------------------------------------------------
vector_2d
geo_conv( vector_2d const& point, int from, int to )
{
  auto const c = s_geo_conv.load();
  if ( !c )
  {
    throw std::runtime_error( "No geo-conversion functor is registered" );
  }

  return ( *c )( point, from, to );
}

// ----------------------------------------------------------------------------
vector_3d
geo_conv( vector_3d const& point, int from, int to )
{
  auto const c = s_geo_conv.load();
  if ( !c )
  {
    throw std::runtime_error( "No geo-conversion functor is registered" );
  }

  return ( *c )( point, from, to );
}

// ----------------------------------------------------------------------------
utm_ups_zone_t
utm_ups_zone( double lon, double lat )
{
  // Check latitude for range error
  if ( lat > 90.0 || lat < -90.0 )
  {
    throw std::range_error( "Input latitude is out of range" );
  }

  // Check for UPS zones
  if ( lat > 84.0 )
  {
    return { 0, true }; // UPS north
  }
  if ( lat < -80.0 )
  {
    return { 0, false }; // UPS south
  }

  // Get normalized longitude and return UTM zone
  lon = fmod( lon, 360.0 );
  auto const zone = 1 + ( ( 30 + static_cast<int>( lon / 6.0 ) ) % 60 );
  return { zone, lat >= 0.0 };
}

// ----------------------------------------------------------------------------
utm_ups_zone_t
utm_ups_zone( vector_2d const& lon_lat)
{
  return utm_ups_zone(lon_lat[0], lon_lat[1]);
}

// ----------------------------------------------------------------------------
utm_ups_zone_t
utm_ups_zone( vector_3d const& lon_lat_alt)
{
  return utm_ups_zone(lon_lat_alt[0], lon_lat_alt[1]);
}

} } // end namespace
