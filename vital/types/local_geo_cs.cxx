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

#include <vital/math_constants.h>

#include <vital/types/geodesy.h>
#include <vital/types/geodesy.h>
#include <vital/types/metadata_traits.h>

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

/// Use the pose data provided by metadata to update camera pose
bool
local_geo_cs
::update_camera(vital::metadata const& md,
                vital::simple_camera_perspective& cam,
                vital::rotation_d const& rot_offset) const
{
  bool rotation_set = false;
  bool translation_set = false;

  bool has_platform_yaw = false;
  bool has_platform_pitch = false;
  bool has_platform_roll = false;
  bool has_sensor_yaw = false;
  bool has_sensor_pitch = false;

  double platform_yaw = 0.0, platform_pitch = 0.0, platform_roll = 0.0;
  if (auto& mdi = md.find(vital::VITAL_META_PLATFORM_HEADING_ANGLE))
  {
    mdi.data(platform_yaw);
    has_platform_yaw = true;
  }
  if (auto& mdi = md.find(vital::VITAL_META_PLATFORM_PITCH_ANGLE))
  {
    mdi.data(platform_pitch);
    has_platform_pitch = true;
  }
  if (auto& mdi = md.find(vital::VITAL_META_PLATFORM_ROLL_ANGLE))
  {
    mdi.data(platform_roll);
    has_platform_roll = true;
  }
  double sensor_yaw = 0.0, sensor_pitch = 0.0, sensor_roll = 0.0;
  if (auto& mdi = md.find(vital::VITAL_META_SENSOR_REL_AZ_ANGLE))
  {
    mdi.data(sensor_yaw);
    has_sensor_yaw = true;
  }
  if (auto& mdi = md.find(vital::VITAL_META_SENSOR_REL_EL_ANGLE))
  {
    mdi.data(sensor_pitch);
    has_sensor_pitch = true;
  }
  if (auto& mdi = md.find(vital::VITAL_META_SENSOR_REL_ROLL_ANGLE))
  {
    mdi.data(sensor_roll);
  }


  if ( has_platform_yaw && has_platform_pitch && has_platform_roll &&
       has_sensor_yaw && has_sensor_pitch &&  // Sensor roll is ignored here on purpose.
                                              // It is fixed on some platforms to zero.
      !(std::isnan(platform_yaw) || std::isnan(platform_pitch) || std::isnan(platform_roll) ||
        std::isnan(sensor_yaw) || std::isnan(sensor_pitch) || std::isnan(sensor_roll)))
  {
    //only set the camera's rotation if all metadata angles are present

    auto R = compose_rotations<double>(platform_yaw, platform_pitch, platform_roll,
                                       sensor_yaw, sensor_pitch, sensor_roll);

    cam.set_rotation(R);

    rotation_set = true;
  }

  if (auto& mdi = md.find(vital::VITAL_META_SENSOR_LOCATION))
  {

    vital::geo_point gloc;
    mdi.data( gloc );

    // get the location in the same UTM zone as the origin
    vector_3d loc = gloc.location(geo_origin_.crs());
    loc -= geo_origin_.location();
    cam.set_center(vector_3d(loc.x(), loc.y(), loc.z()));
    translation_set = true;
  }
  return rotation_set || translation_set;
}

/// Use the camera pose to update the metadata structure
void
local_geo_cs
::update_metadata(vital::simple_camera_perspective const& cam,
                  vital::metadata& md) const
{
  if (md.has(vital::VITAL_META_PLATFORM_HEADING_ANGLE) &&
      md.has(vital::VITAL_META_PLATFORM_PITCH_ANGLE) &&
      md.has(vital::VITAL_META_PLATFORM_ROLL_ANGLE) &&
      md.has(vital::VITAL_META_SENSOR_REL_AZ_ANGLE) &&
      md.has(vital::VITAL_META_SENSOR_REL_EL_ANGLE))
  {  //we have a complete metadata rotation.  Note that sensor roll is ignored here on purpose.
    double yaw, pitch, roll;
    cam.rotation().get_yaw_pitch_roll(yaw, pitch, roll);
    yaw *= rad_to_deg;
    pitch *= rad_to_deg;
    roll *= rad_to_deg;
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_YAW_ANGLE, yaw));
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_PITCH_ANGLE, pitch));
    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_ROLL_ANGLE, roll));
  }

  if (md.has(vital::VITAL_META_SENSOR_LOCATION))
  {
    // we have a complete position from metadata.
    vital::vector_3d c = cam.get_center();
    vital::vector_3d offset = vector_3d(c.x(), c.y(), c.z()) +
      geo_origin_.location();
    vital::geo_point gc( offset, geo_origin_.crs() );

    md.add(NEW_METADATA_ITEM(VITAL_META_SENSOR_LOCATION, gc));
  }
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
