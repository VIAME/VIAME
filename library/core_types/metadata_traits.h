// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief This file contains the interface for metadata traits.

#ifndef KWIVER_VITAL_METADATA_TRAITS_H_
#define KWIVER_VITAL_METADATA_TRAITS_H_

#include <viame/core_types/geo_point.h>
#include <viame/core_types/geo_polygon.h>
#include <viame/core_types/metadata_tags.h>
#include <viame/core_types/rotation.h>
#include <viame/core_types/vital_types_export.h>
#include <viame/core_types/vital_types.h>

#include <typeinfo>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < vital_metadata_tag Tag > struct metadata_tag_static_traits;

#define TAG_TYPE( TAG, TYPE ) \
template <>                   \
struct metadata_tag_static_traits< TAG > { using type = TYPE; }

TAG_TYPE( VITAL_META_UNKNOWN, int );
TAG_TYPE( VITAL_META_METADATA_ORIGIN, string_t );
TAG_TYPE( VITAL_META_UNIX_TIMESTAMP, uint64_t );
TAG_TYPE( VITAL_META_UNIX_NANO_TIMESTAMP, uint64_t );
TAG_TYPE( VITAL_META_UNIX_TIMESTAMP_SOURCE, string_t );
TAG_TYPE( VITAL_META_MISSION_ID, string_t );
TAG_TYPE( VITAL_META_MISSION_NUMBER, string_t );
TAG_TYPE( VITAL_META_PLATFORM_TAIL_NUMBER, string_t );
TAG_TYPE( VITAL_META_PLATFORM_ORIENTATION, rotation_d );
TAG_TYPE( VITAL_META_PLATFORM_TRUE_AIRSPEED, double );
TAG_TYPE( VITAL_META_PLATFORM_INDICATED_AIRSPEED, double );
TAG_TYPE( VITAL_META_PLATFORM_DESIGNATION, string_t );
TAG_TYPE( VITAL_META_IMAGE_SOURCE_SENSOR, string_t );
TAG_TYPE( VITAL_META_IMAGE_COORDINATE_SYSTEM, string_t );
TAG_TYPE( VITAL_META_IMAGE_URI, string_t );
TAG_TYPE( VITAL_META_IMAGE_WIDTH, uint64_t );
TAG_TYPE( VITAL_META_IMAGE_HEIGHT, uint64_t );
TAG_TYPE( VITAL_META_IMAGE_CORRUPT, bool );
TAG_TYPE( VITAL_META_VIDEO_DATA_STREAM_INDEX, int );
TAG_TYPE( VITAL_META_VIDEO_DATA_STREAM_SYNCHRONOUS, bool );
TAG_TYPE( VITAL_META_VIDEO_URI, string_t );
TAG_TYPE( VITAL_META_VIDEO_KEY_FRAME, bool );
TAG_TYPE( VITAL_META_VIDEO_FRAME_NUMBER, uint64_t );
TAG_TYPE( VITAL_META_VIDEO_MICROSECONDS, uint64_t );
TAG_TYPE( VITAL_META_VIDEO_FRAME_RATE, double );
TAG_TYPE( VITAL_META_VIDEO_BITRATE, uint64_t );
TAG_TYPE( VITAL_META_VIDEO_COMPRESSION_TYPE, string_t );
TAG_TYPE( VITAL_META_VIDEO_COMPRESSION_PROFILE, string_t );
TAG_TYPE( VITAL_META_VIDEO_COMPRESSION_LEVEL, string_t );
TAG_TYPE( VITAL_META_SENSOR_LOCATION, geo_point );
TAG_TYPE( VITAL_META_SENSOR_HORIZONTAL_FOV, double );
TAG_TYPE( VITAL_META_SENSOR_VERTICAL_FOV, double );
TAG_TYPE( VITAL_META_SENSOR_ORIENTATION, rotation_d );
TAG_TYPE( VITAL_META_SENSOR_TYPE, string_t );
TAG_TYPE( VITAL_META_SLANT_RANGE, double );
TAG_TYPE( VITAL_META_TARGET_WIDTH, double );
TAG_TYPE( VITAL_META_FRAME_CENTER, geo_point );
TAG_TYPE( VITAL_META_CORNER_POINTS, geo_polygon );
TAG_TYPE( VITAL_META_WIND_DIRECTION, double );
TAG_TYPE( VITAL_META_WIND_SPEED, double );
TAG_TYPE( VITAL_META_STATIC_PRESSURE, double );
TAG_TYPE( VITAL_META_DENSITY_ALTITUDE, double );
TAG_TYPE( VITAL_META_OUTSIDE_AIR_TEMPERATURE, double );
TAG_TYPE( VITAL_META_TARGET_LOCATION, geo_point );
TAG_TYPE( VITAL_META_TARGET_TRK_GATE_WIDTH, double );
TAG_TYPE( VITAL_META_TARGET_TRK_GATE_HEIGHT, double );
TAG_TYPE( VITAL_META_TARGET_ERROR_EST_CE90, double );
TAG_TYPE( VITAL_META_TARGET_ERROR_EST_LE90, double );
TAG_TYPE( VITAL_META_DIFFERENTIAL_PRESSURE, double );
TAG_TYPE( VITAL_META_PLATFORM_ANG_OF_ATTACK, double );
TAG_TYPE( VITAL_META_PLATFORM_VERTICAL_SPEED, double );
TAG_TYPE( VITAL_META_PLATFORM_SIDESLIP_ANGLE, double );
TAG_TYPE( VITAL_META_AIRFIELD_BAROMET_PRESS, double );
TAG_TYPE( VITAL_META_AIRFIELD_ELEVATION, double );
TAG_TYPE( VITAL_META_RELATIVE_HUMIDITY, double );
TAG_TYPE( VITAL_META_PLATFORM_GROUND_SPEED, double );
TAG_TYPE( VITAL_META_GROUND_RANGE, double );
TAG_TYPE( VITAL_META_PLATFORM_FUEL_REMAINING, double );
TAG_TYPE( VITAL_META_PLATFORM_CALL_SIGN, string_t );
TAG_TYPE( VITAL_META_LASER_PRF_CODE, uint64_t );
TAG_TYPE( VITAL_META_SENSOR_FOV_NAME, uint64_t );
TAG_TYPE( VITAL_META_PLATFORM_MAGNET_HEADING, double );
TAG_TYPE( VITAL_META_UAS_LDS_VERSION_NUMBER, uint64_t );
TAG_TYPE( VITAL_META_START_TIMESTAMP, uint64_t );
TAG_TYPE( VITAL_META_EVENT_START_TIMESTAMP, uint64_t );
TAG_TYPE( VITAL_META_SECURITY_CLASSIFICATION, string_t );
TAG_TYPE( VITAL_META_AVERAGE_GSD, double );
TAG_TYPE( VITAL_META_VNIIRS, double );
TAG_TYPE( VITAL_META_WAVELENGTH, string_t );
TAG_TYPE( VITAL_META_GPS_SEC, double );
TAG_TYPE( VITAL_META_GPS_WEEK, int );
TAG_TYPE( VITAL_META_NORTHING_VEL, double );
TAG_TYPE( VITAL_META_EASTING_VEL, double );
TAG_TYPE( VITAL_META_UP_VEL, double );
TAG_TYPE( VITAL_META_IMU_STATUS, int );
TAG_TYPE( VITAL_META_LOCAL_ADJ, int );
TAG_TYPE( VITAL_META_DST_FLAGS, int );
TAG_TYPE( VITAL_META_RPC_HEIGHT_OFFSET, double );
TAG_TYPE( VITAL_META_RPC_HEIGHT_SCALE, double );
TAG_TYPE( VITAL_META_RPC_LONG_OFFSET, double );
TAG_TYPE( VITAL_META_RPC_LONG_SCALE, double );
TAG_TYPE( VITAL_META_RPC_LAT_OFFSET, double );
TAG_TYPE( VITAL_META_RPC_LAT_SCALE, double );
TAG_TYPE( VITAL_META_RPC_ROW_OFFSET, double );
TAG_TYPE( VITAL_META_RPC_ROW_SCALE, double );
TAG_TYPE( VITAL_META_RPC_COL_OFFSET, double );
TAG_TYPE( VITAL_META_RPC_COL_SCALE, double );
TAG_TYPE( VITAL_META_RPC_ROW_NUM_COEFF, string_t );
TAG_TYPE( VITAL_META_RPC_ROW_DEN_COEFF, string_t );
TAG_TYPE( VITAL_META_RPC_COL_NUM_COEFF, string_t );
TAG_TYPE( VITAL_META_RPC_COL_DEN_COEFF, string_t );
TAG_TYPE( VITAL_META_NITF_IDATIM, string_t );
TAG_TYPE( VITAL_META_NITF_BLOCKA_FRFC_LOC_01, string_t );
TAG_TYPE( VITAL_META_NITF_BLOCKA_FRLC_LOC_01, string_t );
TAG_TYPE( VITAL_META_NITF_BLOCKA_LRFC_LOC_01, string_t );
TAG_TYPE( VITAL_META_NITF_BLOCKA_LRLC_LOC_01, string_t );
TAG_TYPE( VITAL_META_NITF_IMAGE_COMMENTS, string_t );

#undef TAG_TYPE

// ----------------------------------------------------------------------------
class VITAL_TYPES_EXPORT metadata_tag_traits
{
public:
  metadata_tag_traits(
    vital_metadata_tag tag, std::string const& enum_name,
    std::type_info const& type, std::string const& name,
    std::string const& description );

  vital_metadata_tag
  tag() const;

  std::string
  name() const;

  std::string
  enum_name() const;

  std::type_info const&
  type() const;

  std::string
  type_name() const;

  std::string
  description() const;

private:
  vital_metadata_tag m_tag;
  std::string m_enum_name;
  std::type_info const& m_type;
  std::string m_name;
  std::string m_description;
};

// ----------------------------------------------------------------------------
VITAL_TYPES_EXPORT
metadata_tag_traits const&
tag_traits_by_tag( vital_metadata_tag tag );

// ----------------------------------------------------------------------------
VITAL_TYPES_EXPORT
metadata_tag_traits const&
tag_traits_by_name( std::string const& name );

// ----------------------------------------------------------------------------
VITAL_TYPES_EXPORT
metadata_tag_traits const&
tag_traits_by_enum_name( std::string const& name );

} // namespace vital

} // namespace kwiver

#endif
