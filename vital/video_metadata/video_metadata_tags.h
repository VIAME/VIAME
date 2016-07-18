/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief This file contains the internal definitions for the vital
 * video metadata tags.
 */

#ifndef KWIVER_VITAL_VIDEO_METADATA_TAGS_H
#define KWIVER_VITAL_VIDEO_METADATA_TAGS_H

//
// Define all VITAL metadata tags
//
// This is the canonical definition for all kwiver vital video metadata flags
//
// Add another line to the macro to add another tag.

//
//      tag                          string name                        type
//      ---                          -----------                        ----
#define KWIVER_VITAL_METADATA_TAGS(CALL)                                \
CALL( UNKNOWN,                     "Unknown / Undefined entry",       void) \
CALL( METADATA_ORIGIN,             "Origin of metadata",              std::string ) \
CALL( UNIX_TIMESTAMP,              "Unix Time Stamp",                 uint64_t) \
CALL( MISSION_ID,                  "Mission ID",                      std::string) \
CALL( MISSION_NUMBER,              "Episode Number",                  std::string) \
CALL( PLATFORM_TAIL_NUMBER,        "Platform Tail Number",            std::string) \
CALL( PLATFORM_HEADING_ANGLE,      "Platform Heading Angle",          double) \
CALL( PLATFORM_PITCH_ANGLE,        "Platform Pitch Angle",            double) \
CALL( PLATFORM_ROLL_ANGLE,         "Platform Roll Angle",             double) \
CALL( PLATFORM_TRUE_AIRSPEED,      "Platform True Airspeed",          double) \
CALL( PLATFORM_INDICATED_AIRSPEED, "Platform Indicated Airspeed",     double) \
CALL( PLATFORM_DESIGNATION,        "Platform Designation",            std::string) \
CALL( IMAGE_SOURCE_SENSOR,         "Image Source Sensor",             std::string) \
CALL( IMAGE_COORDINATE_SYSTEM,     "Image Coordinate System",         std::string) \
CALL( SENSOR_LOCATION,             "Sensor Location Lat/Lon",         geo_lat_lon) \
CALL( SENSOR_ALTITUDE,             "Sensor Altitude",                 double) \
CALL( SENSOR_HORIZONTAL_FOV,       "Sensor Horizontal Field of View", double) \
CALL( SENSOR_VERTICAL_FOV,         "Sensor Vertical Field of View",   double) \
CALL( SENSOR_REL_AZ_ANGLE,         "Sensor Relative Azimuth Angle",   double) \
CALL( SENSOR_REL_EL_ANGLE,         "Sensor Relative Elevation Angle", double) \
CALL( SENSOR_REL_ROLL_ANGLE,       "Sensor Relative Roll Angle",      double) \
CALL( SENSOR_ROLL_ANGLE,           "Sensor Roll Angle",               double) \
CALL( SENSOR_TYPE,                 "Sensor Type",                     std::string) \
CALL( SLANT_RANGE,                 "Slant Range",                     double) \
CALL( TARGET_WIDTH,                "Target Width",                    double) \
CALL( FRAME_CENTER,                "Frame Center Lat/Lon",            geo_lat_lon) \
CALL( FRAME_CENTER_ELEV,           "Frame Center Elevation",          double) \
CALL( CORNER_POINTS,               "Corner points in lat/lon",        geo_corner_points) \
CALL( ICING_DETECTED,              "Icing Detected",                  uint64_t) \
CALL( WIND_DIRECTION,              "Wind Direction",                  double) \
CALL( WIND_SPEED,                  "Wind Speed",                      double) \
CALL( STATIC_PRESSURE,             "Static Pressure",                 double) \
CALL( DENSITY_ALTITUDE,            "Density Altitude",                double) \
CALL( OUTSIDE_AIR_TEMPERATURE,     "Outside Air Temperature",         double) \
CALL( TARGET_LOCATION,             "Target Location Lat/Lon",         geo_lat_lon) \
CALL( TARGET_LOCATION_ELEV,        "Target Location Elevation",       double) \
CALL( TARGET_TRK_GATE_WIDTH,       "Target Track Gate Width",         double) \
CALL( TARGET_TRK_GATE_HEIGHT,      "Target Track Gate Height",        double) \
CALL( TARGET_ERROR_EST_CE90,       "Target Error Estimate - CE90",    double) \
CALL( TARGET_ERROR_EST_LE90,       "Target Error Estimate - LE90",    double) \
CALL( DIFFERENTIAL_PRESSURE,       "Differential Pressure",           double) \
CALL( PLATFORM_ANG_OF_ATTACK,      "Platform Angle of Attack",        double) \
CALL( PLATFORM_VERTICAL_SPEED,     "Platform Vertical Speed",         double) \
CALL( PLATFORM_SIDESLIP_ANGLE,     "Platform Sideslip Angle",         double) \
CALL( AIRFIELD_BAROMET_PRESS,      "Airfield Barometric Pressure",    double) \
CALL( AIRFIELD_ELEVATION,          "Airfield Elevation",              double) \
CALL( RELATIVE_HUMIDITY,           "Relative Humidity",               double) \
CALL( PLATFORM_GROUND_SPEED,       "Platform Ground Speed",           double) \
CALL( GROUND_RANGE,                "Ground Range",                    double) \
CALL( PLATFORM_FUEL_REMAINING,     "Platform Fuel Remaining",         double) \
CALL( PLATFORM_CALL_SIGN,          "Platform Call Sign",              std::string) \
CALL( LASER_PRF_CODE,              "Laser PRF Code",                  uint64_t) \
CALL( SENSOR_FOV_NAME,             "Sensor Field of View Name",       uint64_t) \
CALL( PLATFORM_MAGNET_HEADING,     "Platform Magnetic Heading",       double) \
CALL( UAS_LDS_VERSION_NUMBER,      "UAS LDS Version Number",          uint64_t) \
CALL( ANGLE_TO_NORTH,              "Angle to North",                  double) \
CALL( OBLIQUITY_ANGLE,             "Sensor Elevation Angle",          double) \
CALL( START_DATE_TIME_UTC,         "Start Date Time - UTC",           std::string ) \
CALL( EVENT_START_DATE_TIME_UTC,   "Event Start Date Time - UTC",     std::string ) \
CALL( MISSION_START_TIME_UTC,      "Mission Start Date Time - UTC",   std::string ) \
CALL( SECURITY_CLASSIFICATION,     "Security Classification",         std::string ) \
CALL( CLASSIFICATION,              "Classification (0102 lds)",       std::string ) \
CALL( SECURITY_LOCAL_MD_SET,       "Security Local Metadata Set",     std::string ) /* really std_0102_lds */ \
CALL( 0601_WEAPON_LOAD,            "Weapon Load",                     uint64_t) \
CALL( 0601_WEAPON_FIRED,           "Weapon Fired",                    uint64_t) \
CALL( AVERAGE_GSD,                 "Average GSD value",               double)

// ------------------------------------------------------------------
//
// Canonical metadata tags
//

namespace kwiver {
namespace vital {

enum vital_metadata_tag {

#define ENUM_ITEM( TAG, NAME, T) VITAL_META_ ## TAG,

  // Generate enum items
  KWIVER_VITAL_METADATA_TAGS( ENUM_ITEM )

#undef ENUM_ITEM

  // User tags can be generated for a specific application and
  // should start with a value not less than the following.
  VITAL_META_LAST_TAG
};

} } // end namespace

#endif /* KWIVER_VITAL_VIDEO_METADATA_TAGS_H */
