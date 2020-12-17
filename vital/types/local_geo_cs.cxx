// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
