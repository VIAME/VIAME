/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
utm_ups_zone_t
utm_ups_zone( vector_2d const& lat_lon )
{
  // Get latitude and check for range error
  auto const lat = lat_lon[1];
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
  auto const lon = fmod( lat_lon[0], 360.0 );
  auto const zone = 1 + ( ( 30 + static_cast<int>( lon / 6.0 ) ) % 60 );
  return { zone, lat >= 0.0 };
}

} } // end namespace
