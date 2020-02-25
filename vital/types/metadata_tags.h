/*ckwg +29
 * Copyright 2016-2017, 2019 by Kitware, Inc.
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
 * metadata tags.
 */

#ifndef KWIVER_VITAL_METADATA_TAGS_H_
#define KWIVER_VITAL_METADATA_TAGS_H_

//
// Define all Vital metadata tags
//
// This is the canonical definition for all kwiver vital metadata flags
//
// Add another line to the macro to add another tag.

#define KWIVER_VITAL_METADATA_TAGS( CALL )                              \
  CALL( UNKNOWN,                                                        \
        "Unknown / Undefined entry",                                    \
        int,                                                            \
        "Undefined entry" )                                             \
  CALL( METADATA_ORIGIN,                                                \
        "Origin of metadata",                                           \
        std::string,                                                    \
        "Name of the metadata standard which was the origin of these "  \
        "metadata vaules if "                                           \
        "they originated from a video stream. Can be omitted." )        \
  CALL( UNIX_TIMESTAMP,                                                 \
        "Unix Time Stamp",                                              \
        uint64_t,                                                       \
        "" )                                                            \
  CALL( MISSION_ID,                                                     \
        "Mission ID",                                                   \
        std::string,                                                    \
        "Descriptive Mission Identifier to distinguish event or sortie. "\
        "Value field is Free Text." )                                   \
  CALL( MISSION_NUMBER,                                                 \
        "Episode Number",                                               \
        std::string,                                                    \
        "" )                                                            \
  CALL( PLATFORM_TAIL_NUMBER,                                           \
        "Platform Tail Number",                                         \
        std::string,                                                    \
        "Identifier of platform as posted." )                           \
  CALL( PLATFORM_HEADING_ANGLE,                                         \
        "Platform Heading Angle (deg)",                                 \
        double,                                                         \
        "Aircraft heading angle.  Relative between longitudinal axis "  \
        "and True North measured in the horizontal plane." )            \
  CALL( PLATFORM_PITCH_ANGLE,                                           \
        "Platform Pitch Angle (deg)",                                   \
        double,                                                         \
        "Aircraft pitch angle. Angle between longitudinal axis and "    \
        "horzontal plane.  Positive angles above horizontal plane." )   \
  CALL( PLATFORM_ROLL_ANGLE,                                            \
        "Platform Roll Angle (deg)",                                    \
        double,                                                         \
        "Platform roll angle. Angle between transverse axis and "       \
        "horizontal plane. Positive angles for right wing lowered "     \
        "below horizontal plane." )                                     \
  CALL( PLATFORM_TRUE_AIRSPEED,                                         \
        "Platform True Airspeed (meters/sec)",                          \
        double,                                                         \
        "True airspeed (TAS) of platform. Indicated Airspeed "          \
        "adjusted for temperature and altitude." )                      \
  CALL( PLATFORM_INDICATED_AIRSPEED,                                    \
        "Platform Indicated Airspeed (meters/sec)",                     \
        double,                                                         \
        "Indicated airspeed (IAS) of platform. Derived from Pitot "     \
        "tube and static pressure sensors." )                           \
  CALL( PLATFORM_DESIGNATION,                                           \
        "Platform Designation",                                         \
        std::string,                                                    \
        "" )                                                            \
  CALL( IMAGE_SOURCE_SENSOR,                                            \
        "Image Source Sensor",                                          \
        std::string,                                                    \
        "String of image source sensor.  E.g.: 'EO Nose', "             \
        "'EO Zoom (DLTV)', 'EO Spotter', "                              \
        "'IR Mitsubishi PtSi Model 500', 'IR InSb Amber Model TBT', "   \
        "'LYNX SAR Imagery', 'TESAR Imagery', etc." )                   \
  CALL( IMAGE_COORDINATE_SYSTEM,                                        \
        "Image Coordinate System",                                      \
        std::string,                                                    \
        "Coordinate system used. E.g.: 'Geodetic WGS84', "              \
        "'Geocentric WGS84', 'TUM', 'None', etc." )                     \
  CALL( IMAGE_URI,                                                      \
        "Image URI",                                                    \
        std::string,                                                    \
        "" )                                                            \
  CALL( VIDEO_URI,                                                      \
        "Video URI",                                                    \
        std::string,                                                    \
        "" )                                                            \
  CALL( VIDEO_KEY_FRAME,                                                \
        "Is frame a key frame",                                         \
        bool,                                                           \
        "" )                                                            \
  CALL( SENSOR_LOCATION,                                                \
        "Sensor Geodetic Location (lon/lat/meters)",                    \
        geo_point,                                                      \
        "Contains the 3D coordinate of the sensor. "                    \
        "The location is ordered lon, lat. "                            \
        "The altitude is optional and is in meters." )                  \
  CALL( SENSOR_HORIZONTAL_FOV,                                          \
        "Sensor Horizontal Field of View (deg)",                        \
        double,                                                         \
        "Horizontal field of view of selected imaging sensor." )        \
  CALL( SENSOR_VERTICAL_FOV,                                            \
        "Sensor Vertical Field of View (deg)",                          \
        double,                                                         \
        "Vertical field of view of selected imaging sensor." )          \
  CALL( SENSOR_REL_AZ_ANGLE,                                            \
        "Sensor Relative Azimuth Angle (deg)",                          \
        double,                                                         \
        "Relative rotation angle of sensor to platform longitudinal axis. " \
        "Rotation angle between platform longitudinal axis and " \
        "camera pointing direction as seen from above the platform." )  \
  CALL( SENSOR_REL_EL_ANGLE,                                            \
        "Sensor Relative Elevation Angle (deg)",                        \
        double,                                                         \
        "Relative Elevation Angle of sensor to platform "               \
        "longitudinal-transverse plane. Negative angles down." )        \
  CALL( SENSOR_REL_ROLL_ANGLE,                                          \
        "Sensor Relative Roll Angle (deg)",                             \
        double,                                                         \
        "Relative roll angle of sensor to aircraft platform. "          \
        "Twisting angle of camera about lens axis. "                    \
        "Top of image is zero degrees.  "                               \
        "Positive angles are clockwise when looking from behind camera." ) \
  CALL( SENSOR_YAW_ANGLE,                                               \
        "Sensor yaw angle (deg)",                                       \
        double,                                                         \
        "" )                                                            \
  CALL( SENSOR_PITCH_ANGLE,                                             \
        "Sensor pitch angle (deg)",                                     \
        double,                                                         \
        "" )                                                            \
  CALL( SENSOR_ROLL_ANGLE,                                              \
        "Sensor Roll Angle (deg)",                                      \
        double,                                                         \
        "" )                                                            \
  CALL( SENSOR_TYPE,                                                    \
        "Sensor Type",                                                  \
        std::string,                                                    \
        "" )                                                            \
  CALL( SLANT_RANGE,                                                    \
        "Slant Range (meters)",                                         \
        double,                                                         \
        "Distance to target." )                                         \
  CALL( TARGET_WIDTH,                                                   \
        "Target Width (meters)",                                        \
        double,                                                         \
        "Target Width within sensor field of view." )                   \
  CALL( FRAME_CENTER,                                                   \
        "Geodetic Frame Center, (lon/lat/meters)",                      \
        geo_point,                                                      \
        "Contains the 3D coordinate of the frame center. "              \
        "The location is ordered lon, lat, and altitude in meters. "    \
        "Altitude is not always set." )                                 \
  CALL( CORNER_POINTS,                                                  \
        "Corner points (lon/lat)",                                      \
        geo_polygon,                                                    \
        "A four sided polygon representing the image bounds, "          \
        "The corners are ordered "                                      \
        "upper left, upper right, lower right, lower left.")            \
  CALL( ICING_DETECTED,                                                 \
        "Icing Detected",                                               \
        uint64_t,                                                       \
        "Flag for icing detected at aircraft location." )               \
  CALL( WIND_DIRECTION,                                                 \
        "Wind Direction (deg)",                                         \
        double,                                                         \
        "Wind direction at aircraft location. This is the direction "   \
        "the wind is coming from relative to true north." )             \
  CALL( WIND_SPEED,                                                     \
        "Wind Speed (meters/sec)",                                      \
        double,                                                         \
        "Wind speed at aircraft location." )                            \
  CALL( STATIC_PRESSURE,                                                \
        "Static Pressure (millibar)",                                   \
        double,                                                         \
        "Static pressure at aircraft location." )                       \
  CALL( DENSITY_ALTITUDE,                                               \
        "Density Altitude (meters)",                                    \
        double,                                                         \
        "Density altitude at aircraft location. Relative aircraft "     \
        "performance metric based on outside air temperature, static "  \
        "pressure, and humidity." )                                     \
  CALL( OUTSIDE_AIR_TEMPERATURE,                                        \
        "Outside Air Temperature (Celsius)",                            \
        double,                                                         \
        "Temperature outside aircraft." )                               \
  CALL( TARGET_LOCATION,                                                \
        "Target Geodetic Locationq (lon/lat/meters)",                   \
        geo_point,                                                      \
        "Contains the 3D coordinate of the target. "                    \
        "The location is ordered lon, lat. "                            \
        "The altitude is optional and is in meters." )                  \
  CALL( TARGET_TRK_GATE_WIDTH,                                          \
        "Target Track Gate Width (pixels)",                             \
        double,                                                         \
        "Tracking gate width (x value) of tracked target within field " \
        "of view.  Closely tied to source video resolution in pixels." ) \
  CALL( TARGET_TRK_GATE_HEIGHT,                                         \
        "Target Track Gate Height (pixels)",                            \
        double,                                                         \
        "Tracking gate height (y value) of tracked target within field " \
        "of view.  Closely tied to source video resolution in pixels." ) \
  CALL( TARGET_ERROR_EST_CE90,                                          \
        "Target Error Estimate - CE90 (meters)",                        \
        double,                                                         \
        "Circular Error 90 (CE90) is the estimated error distance "     \
        "in the horizontal direction.  Specifies the radius of 90% "    \
        "probability on a plane tangent to the earth’s surface." )      \
  CALL( TARGET_ERROR_EST_LE90,                                          \
        "Target Error Estimate - LE90 (meters)",                        \
        double,                                                         \
        "Lateral Error 90 (LE90) is the estimated error distance "      \
        "in the vertical (or lateral) direction.  Specifies the "       \
        "interval of 90% probability in the local vertical direction." ) \
  CALL( DIFFERENTIAL_PRESSURE,                                          \
        "Differential Pressure (millibar)",                             \
        double,                                                         \
        "Differential pressure at aircraft location. "                  \
        "Measured as the Stagnation/impact/total pressure minus "       \
        "static pressure." )                                            \
  CALL( PLATFORM_ANG_OF_ATTACK,                                         \
        "Platform Angle of Attack (deg)",                               \
        double,                                                         \
        "Platform Attack Angle. Angle between platform longitudinal "   \
        "axis and relative wind. Positive angles for upward "           \
        "relative wind." )                                              \
  CALL( PLATFORM_VERTICAL_SPEED,                                        \
        "Platform Vertical Speed (meters/sec)",                         \
        double,                                                         \
        "Vertical speed of the aircraft relative to zenith. "           \
        "Positive ascending, negative descending." )                    \
  CALL( PLATFORM_SIDESLIP_ANGLE,                                        \
        "Platform Sideslip Angle (deg)",                                \
        double,                                                         \
        "The sideslip angle is the angle between the platform "         \
        "longitudinal axis and relative wind. "                         \
        "Positive angles to right wing, neg to left." )                 \
  CALL( AIRFIELD_BAROMET_PRESS,                                         \
        "Airfield Barometric Pressure (millibars)",                     \
        double,                                                         \
        "Local pressure at airfield of known height. "                  \
        "Pilot's responsibility to update." )                           \
  CALL( AIRFIELD_ELEVATION,                                             \
        "Airfield Elevation (meters)",                                  \
        double,                                                         \
        "Elevation of Airfield corresponding to Airfield "              \
        "Barometric Pressure." )                                        \
  CALL( RELATIVE_HUMIDITY,                                              \
        "Relative Humidity (percent)",                                  \
        double,                                                         \
        "Relative Humidty at aircraft location." )                      \
  CALL( PLATFORM_GROUND_SPEED,                                          \
        "Platform Ground Speed (meters/sec)",                           \
        double,                                                         \
        "Speed projected to the ground of an airborne platform "        \
        " passing overhead." )                                          \
  CALL( GROUND_RANGE,                                                   \
        "Ground Range (meters)",                                        \
        double,                                                         \
        "Horizontal distance from ground position of aircraft "         \
        "relative to nadir, and target of interest. "                   \
        "Dependent upon Slant Range and `Depression Angle." )           \
  CALL( PLATFORM_FUEL_REMAINING,                                        \
        "Platform Fuel Remaining (Kilogram)",                           \
        double,                                                         \
        "Remainging fuel on airborne platform. "                        \
        "Metered as fuel weight remaining." )                           \
  CALL( PLATFORM_CALL_SIGN,                                             \
        "Platform Call Sign",                                           \
        std::string,                                                    \
        "Call Sign of platform or operating unit." )                    \
  CALL( LASER_PRF_CODE,                                                 \
        "Laser PRF Code",                                               \
        uint64_t,                                                       \
        "A laser's Pulse Repetition Frequency (PRF) code used to "      \
        "mark a target. The Laser PRF code is a three or four digit "   \
        "number consisting of the values 1..8." )                       \
  CALL( SENSOR_FOV_NAME,                                                \
        "Sensor Field of View Name",                                    \
        uint64_t,                                                       \
        "Names sensor field of view quantized steps. "                  \
        "00 = Ultranarrow; 01 = Narrow; 02 = Medium; 03 = Wide; "       \
        "04 = Ultrawide; 05 = Narrow Medium; 06 = 2x Ultranarrow; "     \
        "07 = 4x Ultranarrow" )                                         \
  CALL( PLATFORM_MAGNET_HEADING,                                        \
        "Platform Magnetic Heading (deg)",                              \
        double,                                                         \
        "Aircraft magnetic heading angle. Relative between "            \
        "longitudinal axis and Magnetic North measured in the "         \
        "horizontal plane." )                                           \
  CALL( UAS_LDS_VERSION_NUMBER,                                         \
        "UAS LDS Version Number",                                       \
        uint64_t,                                                       \
        "" )                                                            \
  CALL( ANGLE_TO_NORTH,                                                 \
        "Angle to North (deg)",                                         \
        double,                                                         \
        "" )                                                            \
  CALL( OBLIQUITY_ANGLE,                                                \
        "Sensor Elevation Angle (deg)",                                 \
        double,                                                         \
        "" )                                                            \
  CALL( START_DATE_TIME_UTC,                                            \
        "Start Date Time (UTC)",                                        \
        std::string,                                                    \
        "" )                                                            \
  CALL( EVENT_START_DATE_TIME_UTC,                                      \
        "Event Start Date Time (UTC)",                                  \
        std::string,                                                    \
        "" )                                                            \
  CALL( MISSION_START_TIME_UTC,                                         \
        "Mission Start Date Time (UTC)",                                \
        std::string,                                                    \
        "" )                                                            \
  CALL( SECURITY_CLASSIFICATION,                                        \
        "Security Classification",                                      \
        std::string,                                                    \
        "" )                                                            \
  CALL( CLASSIFICATION,                                                 \
        "Classification (0102 lds)",                                    \
        std::string,                                                    \
        "" )                                                            \
  CALL( SECURITY_LOCAL_MD_SET,                                          \
        "Security Local Metadata Set",                                  \
        std::string,                                                    \
        "Refer to std 0102 lds" )                                       \
  CALL( 0601_WEAPON_LOAD,                                               \
        "Weapon Load",                                                  \
        uint64_t,                                                       \
        "Current weapons stored on aircraft" )                          \
  CALL( 0601_WEAPON_FIRED,                                              \
        "Weapon Fired",                                                 \
        uint64_t,                                                       \
        "Indication when a particular weapon is released. "             \
        "Correlate with Unix Time stamp. "                              \
        "Identical format to Weapon Load." )                            \
  CALL( AVERAGE_GSD,                                                    \
        "Average GSD value (meters/pixel)",                             \
        double,                                                         \
        "" )                                                            \
  CALL( GPS_SEC,                                                        \
        "GPS seconds",                                                  \
        double,                                                         \
        "" )                                                            \
  CALL( GPS_WEEK,                                                       \
        "GPS week",                                                     \
        int,                                                            \
        "" )                                                            \
  CALL( NORTHING_VEL,                                                   \
        "Northing velocity (meters/sec)",                               \
        double,                                                         \
        "" )                                                            \
  CALL( EASTING_VEL,                                                    \
        "Easting velocity (meters/sec)",                                \
        double,                                                         \
        "" )                                                            \
  CALL( UP_VEL,                                                         \
        "UP velocity (meters/sec)",                                     \
        double,                                                         \
        "" )                                                            \
  CALL( IMU_STATUS,                                                     \
        "IMU status",                                                   \
        int,                                                            \
        "" )                                                            \
  CALL( LOCAL_ADJ,                                                      \
        "Local adj",                                                    \
        int,                                                            \
        "" )                                                            \
  CALL( DST_FLAGS,                                                      \
        "Dst flags",                                                    \
        int,                                                            \
        "" )                                                            \
  CALL( RPC_HEIGHT_OFFSET,                                              \
        "RPC height offset",                                            \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_HEIGHT_SCALE,                                               \
        "RPC height scale",                                             \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_LONG_OFFSET,                                                \
        "RPC longitude offset",                                         \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_LONG_SCALE,                                                 \
        "RPC longitude scale",                                          \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_LAT_OFFSET,                                                 \
        "RPC latitude offset",                                          \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_LAT_SCALE,                                                  \
        "RPC latitude scale",                                           \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_ROW_OFFSET,                                                 \
        "RPC row offset",                                               \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_ROW_SCALE,                                                  \
        "RPC row scale",                                                \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_COL_OFFSET,                                                 \
        "RPC column offset",                                            \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_COL_SCALE,                                                  \
        "RPC column scale",                                             \
        double,                                                         \
        "" )                                                            \
  CALL( RPC_ROW_NUM_COEFF,                                              \
        "RPC row numerator coefficients",                               \
        std::string,                                                    \
        "" )                                                            \
  CALL( RPC_ROW_DEN_COEFF,                                              \
        "RPC row denominator coefficients",                             \
        std::string,                                                    \
        "" )                                                            \
  CALL( RPC_COL_NUM_COEFF,                                              \
        "RPC column numerator coefficients",                            \
        std::string,                                                    \
        "" )                                                            \
  CALL( RPC_COL_DEN_COEFF,                                              \
        "RPC column denominator coefficients",                          \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_IDATIM,                                                    \
        "NITF IDATIM",                                                  \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_BLOCKA_FRFC_LOC_01,                                        \
        "First Row First Column Location",                              \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_BLOCKA_FRLC_LOC_01,                                        \
        "First Row Last Column Location",                               \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_BLOCKA_LRLC_LOC_01,                                        \
        "Last Row Last Column Location",                                \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_BLOCKA_LRFC_LOC_01,                                        \
        "Last Row First Column Location",                               \
        std::string,                                                    \
        "" )                                                            \
  CALL( NITF_IMAGE_COMMENTS,                                            \
        "Image Comments for NITF File",                                 \
        std::string,                                                    \
        "" )

// ------------------------------------------------------------------
//
// Canonical metadata tags
//

namespace kwiver {
namespace vital {

enum vital_metadata_tag {

#define ENUM_ITEM(TAG, NAME, T, ...) VITAL_META_ ## TAG,

  // Generate enum items
  KWIVER_VITAL_METADATA_TAGS( ENUM_ITEM )

#undef ENUM_ITEM

  // User tags can be generated for a specific application and
  // should start with a value not less than the following.
  VITAL_META_LAST_TAG
};

} } // end namespace

#endif
