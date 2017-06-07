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
 * \brief This file contains base types and structures for geodesy.
 */

#ifndef KWIVER_VITAL_GEODESY_H_
#define KWIVER_VITAL_GEODESY_H_

#include <vital/vital_config.h>
#include <vital/vital_export.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/** Well known coordinate reference systems.
 *
 * This enumeration provides a set of well known coordinate reference systems
 * (CRS's). The numeric values correspond to geodetic CRS's as specified by
 * the European Petroleum Survey Group (EPSG) Spatial Reference System
 * Identifiers (SRID).
 *
 * \note UTM SRID's are obtained by adding the UTM zone number to the base
 *       SRID
 *
 * \see https://en.wikipedia.org/wiki/Spatial_reference_system,
 *      http://www.epsg.org/, https://epsg-registry.org/
 */
enum class SRID : int
{
  lat_lon_NAD83 = 4269,
  lat_lon_WGS84 = 4326,
  UTM_WGS84_north = 32600, // Add zone number to get zoned SRID
  UTM_WGS84_south = 32700, // Add zone number to get zoned SRID
  UTM_NAD83_northeast = 3313, // Add zone number (59N - 60N) to get zoned SRID
  UTM_NAD83_northwest = 26900, // Add zone number (1N - 23N) to get zoned SRID
};


} } // end namespace

#endif /* KWIVER_VITAL_GEODESY_H_ */
