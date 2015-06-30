/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef _KWIVER_GEO_LAT_LON_H_
#define _KWIVER_GEO_LAT_LON_H_

#include <kwiver/kwiver_export.h>

#include <ostream>

namespace kwiver
{

// ----------------------------------------------------------------
/** Geo-coordinate as latitude / longitude.
 *
 * This class represents a Latitude / longitude geolocated point.
 *
 * The values for latitude and longitude are in degrees, but there is
 * no required convention for longitude. It can be either 0..360 or
 * -180 .. 0 .. 180. It is up to the application.  If a specific
 * convention must be enforced, make a subclass.
 */
class KWIVER_EXPORT geo_lat_lon
{
public:
  static const double INVALID;    // used to indicate uninitialized value

  geo_lat_lon();
  geo_lat_lon( double lat, double lon );

  virtual ~geo_lat_lon();

  geo_lat_lon& set_latitude( double l );
  geo_lat_lon& set_longitude( double l );
  double get_latitude() const;
  double get_longitude() const;

  bool is_empty() const;
  bool is_valid() const;

  bool operator==( const geo_lat_lon& rhs ) const;
  bool operator!=( const geo_lat_lon& rhs ) const;

private:

  double m_latitude;
  double m_longitude;
};

KWIVER_EXPORT std::ostream& operator<< (std::ostream& str, kwiver::geo_lat_lon const& obj);

} // end namespace

#endif /* _KWIVER_GEO_LAT_LON_H_ */
