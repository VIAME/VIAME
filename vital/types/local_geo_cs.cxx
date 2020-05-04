/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
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
 * \brief local_geo_cs implementation
 */

#include "local_geo_cs.h"

#include <fstream>
#include <iomanip>

#include <vital/types/geodesy.h>

using namespace kwiver::vital;

namespace kwiver {
namespace vital {

/// Constructor
local_geo_cs
::local_geo_cs()
: geo_origin_()
{
}


/// Set the geographic coordinate origin
void
local_geo_cs
::set_origin(const geo_point& origin)
{
  // convert the origin point into WGS84 UTM for the appropriate zone
  vector_3d lon_lat_alt = origin.location( SRID::lat_lon_WGS84 );
  auto zone = utm_ups_zone( lon_lat_alt );
  int crs = (zone.north ? SRID::UTM_WGS84_north : SRID::UTM_WGS84_south) + zone.number;
  geo_origin_ = geo_point(origin.location(crs), crs);
}

/// Read a local_geo_cs from a text file
void
read_local_geo_cs_from_file(local_geo_cs& lgcs,
                            vital::path_t const& file_path)
{
  std::ifstream ifs(file_path);
  double lat, lon, alt;
  ifs >> lat >> lon >> alt;
  lgcs.set_origin( geo_point( vector_3d(lon, lat, alt), SRID::lat_lon_WGS84) );
}

/// Write a local_geo_cs to a text file
void
write_local_geo_cs_to_file(local_geo_cs const& lgcs,
                           vital::path_t const& file_path)
{
  // write out the origin of the local coordinate system
  auto lon_lat_alt = lgcs.origin().location( SRID::lat_lon_WGS84 );
  std::ofstream ofs(file_path);
  if (ofs)
  {
    ofs << std::setprecision(12) << lon_lat_alt[1]
                                 << " " << lon_lat_alt[0]
                                 << " " << lon_lat_alt[2];
  }
}

} // end namespace vital
} // end namespace kwiver
