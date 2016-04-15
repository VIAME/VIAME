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
 * \brief This file contains the interface to a lat lon geo point.
 */

#ifndef KWIVER_VITAL_GEO_LAT_LON_H_
#define KWIVER_VITAL_GEO_LAT_LON_H_

#include <vital/vital_export.h>

#include <ostream>

namespace kwiver {
namespace vital {

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
class VITAL_EXPORT geo_lat_lon
{
public:
  static const double INVALID;    // used to indicate uninitialized value

  geo_lat_lon();
  geo_lat_lon( double lat, double lon );

  virtual ~geo_lat_lon();

  geo_lat_lon& set_latitude( double l );
  geo_lat_lon& set_longitude( double l );
  double latitude() const;
  double longitude() const;

  /**
   * @brief Is geo location in initial state
   *
   * This method checks the object to see if any data values have been
   * set.
   *
   * @return \b true if object is default constructed.
   */
  bool is_empty() const;

  /**
   * @brief Is geo location a valid coordinate.
   *
   * This method checks the latitude to be between -90 and +90.  The
   * latitude is checked to be between -180 and +360.
   *
   * @return \b true if valid lat/lon coordinate.
   */
  bool is_valid() const;

  bool operator==( const geo_lat_lon& rhs ) const;
  bool operator!=( const geo_lat_lon& rhs ) const;

private:

  double m_latitude;
  double m_longitude;
};

VITAL_EXPORT std::ostream& operator<< (std::ostream& str, vital::geo_lat_lon const& obj);

} } // end namespace

#endif /* KWIVER_VITAL_GEO_LAT_LON_H_ */
