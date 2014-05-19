/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "geo_lat_lon.h"
#include <cmath>
#include <iomanip>

namespace kwiver
{

const double geo_lat_lon::INVALID = 444.0;

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


geo_lat_lon& geo_lat_lon
::set_latitude(double l)
{
  m_latitude = l;
  return ( *this );
}


geo_lat_lon& geo_lat_lon
::set_longitude(double l)
{
  m_longitude = l;
  return ( *this );
}


double geo_lat_lon
::get_latitude() const
{
  return ( m_latitude );
}


double geo_lat_lon
::get_longitude() const
{
  return ( m_longitude );
}


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


bool
geo_lat_lon::
is_empty() const
{
  return (INVALID == get_latitude() && INVALID == get_longitude());
}


bool
geo_lat_lon::
operator == ( const geo_lat_lon &rhs ) const
{
  return ( ( rhs.get_latitude() == this->get_latitude() )
           && ( rhs.get_longitude() == this->get_longitude() ) );
}


bool
geo_lat_lon::
operator != ( const geo_lat_lon &rhs ) const
{
  return ( !( this->operator == ( rhs ) ) );
}


  std::ostream & operator<< (std::ostream & str, kwiver::geo_lat_lon const& obj)
{
  std::streamsize old_prec = str.precision();

  str << std::setprecision(22)
      << "[ " << obj.get_latitude()
      << " / " << obj.get_longitude()
      << " ]";

  str.precision( old_prec );
  return str;
}

} // end namespace
