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
 * \brief This file contains the interface to a geo point.
 */

#ifndef KWIVER_VITAL_GEO_POINT_H_
#define KWIVER_VITAL_GEO_POINT_H_

#include <vital/vital_config.h>
#include <vital/vital_export.h>
#include <vital/types/vector.h>

#include <unordered_map>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/** Geo-coordinate.
 *
 * This class represents a geolocated point. The point is created by specifying
 * a raw location and a CRS. The original location and original CRS may be
 * directly accessed, or the location in a specific CRS may be requested.
 * Requests for a specific CRS are cached, so that CRS conversion does not need
 * to be performed every time.
 */
class VITAL_EXPORT geo_point
{
public:
  using geo_raw_point_t = kwiver::vital::vector_2d;

  geo_point();
  geo_point( geo_raw_point_t const&, int crs );

  virtual ~geo_point() VITAL_DEFAULT_DTOR

  /**
   * \throws std::out_of_range if no location has been set.
   */
  geo_raw_point_t location() const;
  int crs() const;

  /**
   * \throws std::runtime_error if the conversion fails.
   */
  geo_raw_point_t location( int crs ) const;

  void set_location( geo_raw_point_t const&, int crs );

  /**
   * @brief Does point have a specified location.
   *
   * This method checks the object to see if any location data has been set.
   *
   * @return \b true if object is default constructed.
   */
  bool is_empty() const;


  bool operator==( const geo_point& rhs ) const;
  bool operator!=( const geo_point& rhs ) const;

protected:

  int m_original_crs;
  mutable std::unordered_map< int, geo_raw_point_t > m_loc;
};

VITAL_EXPORT std::ostream& operator<< (std::ostream& str, vital::geo_point const& obj);

} } // end namespace

#endif /* KWIVER_VITAL_GEO_POINT_H_ */
