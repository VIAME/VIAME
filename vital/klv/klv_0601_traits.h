/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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
 * \brief This file contains the internal definitions needed to
 * make KLV 0601 traits.
 */

#ifndef KWIVER_VITAL_KLV_0601_TRAITS_H_
#define KWIVER_VITAL_KLV_0601_TRAITS_H_

#include <vital/klv/klv_0601.h>
#include <limits>

namespace kwiver {
namespace vital {

class std_0102_lds { };


/// Define traits for a given KLV 0601 tag
/// All tag traits should be defined using the macro below
/// The anonymous enums allow the values available to the
/// compiler to for use in generic programming.  For values
/// that are already enums, a function is provided so the
/// user does not need to worry about enum type clashes.
template <klv_0601_tag tag>
struct klv_0601_traits;

#define KLV_TRAITS(TAG, NAME, T)                                        \
template <>                                                             \
struct klv_0601_traits<KLV_0601_##TAG>                                  \
{                                                                       \
  static inline std::string name() { return NAME; }                     \
  typedef T type;                                                       \
  static inline klv_0601_tag tag_value() { return KLV_0601_##TAG; }     \
}

//          tag                          string name                        type
//          ---                          -----------                        ----
KLV_TRAITS( CHECKSUM,                    "Checksum",                        uint16_t);
KLV_TRAITS( UNIX_TIMESTAMP,              "Unix Time Stamp",                 uint64_t);
KLV_TRAITS( MISSION_ID,                  "Mission ID",                      std::string);
KLV_TRAITS( PLATFORM_TAIL_NUMBER,        "Platform Tail Number",            std::string);
KLV_TRAITS( PLATFORM_HEADING_ANGLE,      "Platform Heading Angle",          uint16_t);
KLV_TRAITS( PLATFORM_PITCH_ANGLE,        "Platform Pitch Angle",            int16_t);
KLV_TRAITS( PLATFORM_ROLL_ANGLE,         "Platform Roll Angle",             int16_t);
KLV_TRAITS( PLATFORM_TRUE_AIRSPEED,      "Platform True Airspeed",          uint8_t);
KLV_TRAITS( PLATFORM_INDICATED_AIRSPEED, "Platform Indicated Airspeed",     uint8_t);
KLV_TRAITS( PLATFORM_DESIGNATION,        "Platform Designation",            std::string);
KLV_TRAITS( IMAGE_SOURCE_SENSOR,         "Image Source Sensor",             std::string);
KLV_TRAITS( IMAGE_COORDINATE_SYSTEM,     "Image Coordinate System",         std::string);
KLV_TRAITS( SENSOR_LATITUDE,             "Sensor Latitude",                 int32_t);
KLV_TRAITS( SENSOR_LONGITUDE,            "Sensor Longitude",                int32_t);
KLV_TRAITS( SENSOR_TRUE_ALTITUDE,        "Sensor True Altitude",            uint16_t);
KLV_TRAITS( SENSOR_HORIZONTAL_FOV,       "Sensor Horizontal Field of View", uint16_t);
KLV_TRAITS( SENSOR_VERTICAL_FOV,         "Sensor Vertical Field of View",   uint16_t);
KLV_TRAITS( SENSOR_REL_AZ_ANGLE,         "Sensor Relative Azimuth Angle",   uint32_t);
KLV_TRAITS( SENSOR_REL_EL_ANGLE,         "Sensor Relative Elevation Angle", int32_t);
KLV_TRAITS( SENSOR_REL_ROLL_ANGLE,       "Sensor Relative Roll Angle",      uint32_t);
KLV_TRAITS( SLANT_RANGE,                 "Slant Range",                     uint32_t);
KLV_TRAITS( TARGET_WIDTH,                "Target Width",                    uint16_t);
KLV_TRAITS( FRAME_CENTER_LAT,            "Frame Center Latitude",           int32_t);
KLV_TRAITS( FRAME_CENTER_LONG,           "Frame Center Longitude",          int32_t);
KLV_TRAITS( FRAME_CENTER_ELEV,           "Frame Center Elevation",          uint16_t);
KLV_TRAITS( OFFSET_CORNER_LAT_PT_1,      "Offset Corner Latitude Point 1",  int16_t);
KLV_TRAITS( OFFSET_CORNER_LONG_PT_1,     "Offset Corner Longitude Point 1", int16_t);
KLV_TRAITS( OFFSET_CORNER_LAT_PT_2,      "Offset Corner Latitude Point 2",  int16_t);
KLV_TRAITS( OFFSET_CORNER_LONG_PT_2,     "Offset Corner Longitude Point 2", int16_t);
KLV_TRAITS( OFFSET_CORNER_LAT_PT_3,      "Offset Corner Latitude Point 3",  int16_t);
KLV_TRAITS( OFFSET_CORNER_LONG_PT_3,     "Offset Corner Longitude Point 3", int16_t);
KLV_TRAITS( OFFSET_CORNER_LAT_PT_4,      "Offset Corner Latitude Point 4",  int16_t);
KLV_TRAITS( OFFSET_CORNER_LONG_PT_4,     "Offset Corner Longitude Point 4", int16_t);
KLV_TRAITS( ICING_DETECTED,              "Icing Detected",                  uint8_t);
KLV_TRAITS( WIND_DIRECTION,              "Wind Direction",                  uint16_t);
KLV_TRAITS( WIND_SPEED,                  "Wind Speed",                      uint8_t);
KLV_TRAITS( STATIC_PRESSURE,             "Static Pressure",                 uint16_t);
KLV_TRAITS( DENSITY_ALTITUDE,            "Density Altitude",                uint16_t);
KLV_TRAITS( OUTSIDE_AIR_TEMPERATURE,     "Outside Air Temperature",         int8_t);
KLV_TRAITS( TARGET_LOCATION_LAT,         "Target Location Latitude",        int32_t);
KLV_TRAITS( TARGET_LOCATION_LONG,        "Target Location Longitude",       int32_t);
KLV_TRAITS( TARGET_LOCATION_ELEV,        "Target Location Elevation",       uint16_t);
KLV_TRAITS( TARGET_TRK_GATE_WIDTH,       "Target Track Gate Width",         uint8_t);
KLV_TRAITS( TARGET_TRK_GATE_HEIGHT,      "Target Track Gate Height",        uint8_t);
KLV_TRAITS( TARGET_ERROR_EST_CE90,       "Target Error Estimate - CE90",    uint16_t);
KLV_TRAITS( TARGET_ERROR_EST_LE90,       "Target Error Estimate - LE90",    uint16_t);
KLV_TRAITS( GENERIC_FLAG_DATA_01,        "Generic Flag Data 01",            uint8_t);
KLV_TRAITS( SECURITY_LOCAL_MD_SET,       "Security Local Metadata Set",     std_0102_lds);
KLV_TRAITS( DIFFERENTIAL_PRESSURE,       "Differential Pressure",           uint16_t);
KLV_TRAITS( PLATFORM_ANG_OF_ATTACK,      "Platform Angle of Attack",        int16_t);
KLV_TRAITS( PLATFORM_VERTICAL_SPEED,     "Platform Vertical Speed",         int16_t);
KLV_TRAITS( PLATFORM_SIDESLIP_ANGLE,     "Platform Sideslip Angle",         int16_t);
KLV_TRAITS( AIRFIELD_BAROMET_PRESS,      "Airfield Barometric Pressure",    uint16_t);
KLV_TRAITS( AIRFIELD_ELEVATION,          "Airfield Elevation",              uint16_t);
KLV_TRAITS( RELATIVE_HUMIDITY,           "Relative Humidity",               uint8_t);
KLV_TRAITS( PLATFORM_GROUND_SPEED,       "Platform Ground Speed",           uint8_t);
KLV_TRAITS( GROUND_RANGE,                "Ground Range",                    uint32_t);
KLV_TRAITS( PLATFORM_FUEL_REMAINING,     "Platform Fuel Remaining",         uint16_t);
KLV_TRAITS( PLATFORM_CALL_SIGN,          "Platform Call Sign",              std::string);
KLV_TRAITS( WEAPON_LOAD,                 "Weapon Load",                     uint16_t);
KLV_TRAITS( WEAPON_FIRED,                "Weapon Fired",                    uint8_t);
KLV_TRAITS( LASER_PRF_CODE,              "Laser PRF Code",                  uint16_t);
KLV_TRAITS( SENSOR_FOV_NAME,             "Sensor Field of View Name",       uint8_t);
KLV_TRAITS( PLATFORM_MAGNET_HEADING,     "Platform Magnetic Heading",       uint16_t);
KLV_TRAITS( UAS_LDS_VERSION_NUMBER,      "UAS LDS Version Number",          uint16_t);

#undef KLV_TRAITS

//
// These converters are templated over the tags and provide tag
// specific conversion operations.
//
// Default converter.
//
template <klv_0601_tag tag>
struct klv_0601_convert
{
  static const bool has_double = false;
  typedef typename klv_0601_traits<tag>::type type;
  static inline double as_double(const type&)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
};

#define KLV_CAST(TAG)                                   \
template <>                                             \
struct klv_0601_convert<KLV_0601_##TAG>                 \
{                                                       \
  static const bool has_double = true;                  \
  typedef klv_0601_traits<KLV_0601_##TAG>::type type;   \
  static inline double as_double(const type& val)       \
  {                                                     \
    return static_cast<double>(val);                    \
  }                                                     \
}

#define KLV_SCALE(TAG, SCALE)                                           \
template <>                                                             \
struct klv_0601_convert<KLV_0601_##TAG>                                 \
{                                                                       \
  static const bool has_double = true;                                  \
  typedef klv_0601_traits<KLV_0601_##TAG>::type type;                   \
  static inline double as_double(const type& val)                       \
  {                                                                     \
    return (static_cast<double>(val) * SCALE) / std::numeric_limits<type>::max(); \
  }                                                                     \
}

#define KLV_SCALE_OFFSET(TAG, SCALE, OFFSET)                            \
template <>                                                             \
struct klv_0601_convert<KLV_0601_##TAG>                                 \
{                                                                       \
  static const bool has_double = true;                                  \
  typedef klv_0601_traits<KLV_0601_##TAG>::type type;                   \
  static inline double as_double(const type& val)                       \
  {                                                                     \
    return (static_cast<double>(val) * SCALE) / std::numeric_limits<type>::max() + OFFSET; \
  }                                                                     \
}

#define KLV_SCALE_INVALID(TAG, SCALE)                                   \
template <>                                                             \
struct klv_0601_convert<KLV_0601_##TAG>                                 \
{                                                                       \
  static const bool has_double = true;                                  \
  typedef klv_0601_traits<KLV_0601_##TAG>::type type;                   \
  static inline double as_double(const type& val)                       \
  {                                                                     \
    return (val == std::numeric_limits<type>::min())                    \
           ? std::numeric_limits<double>::quiet_NaN()                   \
           : (static_cast<double>(val) * SCALE) / std::numeric_limits<type>::max(); \
  }                                                                     \
}

//                 tag                            scale  offset
//                 ---                            -----  ------
KLV_SCALE(         PLATFORM_HEADING_ANGLE,       360);
KLV_SCALE_INVALID( PLATFORM_PITCH_ANGLE,         20);
KLV_SCALE_INVALID( PLATFORM_ROLL_ANGLE,          50);
KLV_SCALE(         PLATFORM_TRUE_AIRSPEED,       255);
KLV_SCALE(         PLATFORM_INDICATED_AIRSPEED,  255);
KLV_SCALE_INVALID( SENSOR_LATITUDE,              90);
KLV_SCALE_INVALID( SENSOR_LONGITUDE,             180);
KLV_SCALE_OFFSET(  SENSOR_TRUE_ALTITUDE,         19900,  -900);
KLV_SCALE(         SENSOR_HORIZONTAL_FOV,        180);
KLV_SCALE(         SENSOR_VERTICAL_FOV,          180);
KLV_SCALE(         SENSOR_REL_AZ_ANGLE,          360);
KLV_SCALE_INVALID( SENSOR_REL_EL_ANGLE,          180);
KLV_SCALE(         SENSOR_REL_ROLL_ANGLE,        360);
KLV_SCALE(         SLANT_RANGE,                  5000000);
KLV_SCALE(         TARGET_WIDTH,                 10000);
KLV_SCALE_INVALID( FRAME_CENTER_LAT,             90);
KLV_SCALE_INVALID( FRAME_CENTER_LONG,            180);
KLV_SCALE_OFFSET(  FRAME_CENTER_ELEV,            19900,  -900);
KLV_SCALE_INVALID( OFFSET_CORNER_LAT_PT_1,       0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LONG_PT_1,      0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LAT_PT_2,       0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LONG_PT_2,      0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LAT_PT_3,       0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LONG_PT_3,      0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LAT_PT_4,       0.075);
KLV_SCALE_INVALID( OFFSET_CORNER_LONG_PT_4,      0.075);
KLV_SCALE(         WIND_DIRECTION,               360);
KLV_SCALE(         WIND_SPEED,                   100);
KLV_SCALE(         STATIC_PRESSURE,              5000);
KLV_SCALE_OFFSET(  DENSITY_ALTITUDE,             19900,  -900);
KLV_SCALE_INVALID( TARGET_LOCATION_LAT,          90);
KLV_SCALE_INVALID( TARGET_LOCATION_LONG,         180);
KLV_SCALE_OFFSET(  TARGET_LOCATION_ELEV,         19900,  -900);
KLV_CAST(          TARGET_TRK_GATE_WIDTH );
KLV_CAST(          TARGET_TRK_GATE_HEIGHT );
KLV_CAST(          TARGET_ERROR_EST_CE90 );
KLV_CAST(          TARGET_ERROR_EST_LE90 );
KLV_SCALE(         DIFFERENTIAL_PRESSURE,        5000);
KLV_SCALE_INVALID( PLATFORM_ANG_OF_ATTACK,       20);
KLV_SCALE_INVALID( PLATFORM_VERTICAL_SPEED,      180);
KLV_SCALE_INVALID( PLATFORM_SIDESLIP_ANGLE,      20);
KLV_SCALE(         AIRFIELD_BAROMET_PRESS,       5000);
KLV_SCALE_OFFSET(  AIRFIELD_ELEVATION,           19900, -900);
KLV_SCALE(         RELATIVE_HUMIDITY,            100);
KLV_CAST(          PLATFORM_GROUND_SPEED );
KLV_SCALE(         GROUND_RANGE,                 5000000);
KLV_SCALE(         PLATFORM_FUEL_REMAINING,      10000);
KLV_SCALE(         PLATFORM_MAGNET_HEADING,      360);
KLV_CAST(          UAS_LDS_VERSION_NUMBER );

#undef KLV_SCALE
#undef KLV_CAST
#undef KLV_SCALE_OFFSET
#undef KLV_SCALE_INVALID

} } // end namespace

#endif
