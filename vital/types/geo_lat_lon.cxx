/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief This file contains the implementation of a lat lon geo point.
 */

#include "geo_lat_lon.h"
#include <cmath>
#include <iomanip>

namespace kwiver {
namespace vital {

const double geo_lat_lon::INVALID = 444.0;

// ------------------------------------------------------------------
geo_lat_lon::
geo_lat_lon()
  : m_latitude(INVALID),
    m_longitude(INVALID)
{ }

geo_lat_lon::
geo_lat_lon(double lat, double lon)
  : m_latitude(lat),
    m_longitude(lon)
{ }


geo_lat_lon::
~geo_lat_lon()
{ }


// ------------------------------------------------------------------
geo_lat_lon& geo_lat_lon
::set_latitude(double l)
{
  m_latitude = l;
  return ( *this );
}


// ------------------------------------------------------------------
geo_lat_lon& geo_lat_lon
::set_longitude(double l)
{
  m_longitude = l;
  return ( *this );
}


// ------------------------------------------------------------------
double geo_lat_lon
::latitude() const
{
  return ( m_latitude );
}


// ------------------------------------------------------------------
double geo_lat_lon
::longitude() const
{
  return ( m_longitude );
}


// ------------------------------------------------------------------
bool
geo_lat_lon::
is_valid() const
{
  bool valid = true;
  if (!(m_latitude >= -90 && m_latitude <= 90))
  {
    valid = false;
  }
  else if (!(m_longitude >= -180 && m_longitude <= 360))
  {
    valid = false;
  }

  return valid;
}


// ------------------------------------------------------------------
bool
geo_lat_lon::
is_empty() const
{
  return (INVALID == latitude() && INVALID == longitude());
}


// ------------------------------------------------------------------
bool
geo_lat_lon::
operator == ( const geo_lat_lon &rhs ) const
{
  return ( ( rhs.latitude() == this->latitude() )
           && ( rhs.longitude() == this->longitude() ) );
}


// ------------------------------------------------------------------
bool
geo_lat_lon::
operator != ( const geo_lat_lon &rhs ) const
{
  return ( !( this->operator == ( rhs ) ) );
}


// ------------------------------------------------------------------
std::ostream & operator<< (std::ostream & str, vital::geo_lat_lon const& obj)
{
  std::streamsize old_prec = str.precision();

  str << std::setprecision(22)
      << "[ " << obj.latitude()
      << " / " << obj.longitude()
      << " ]";

  str.precision( old_prec );
  return str;
}

} } // end namespace
