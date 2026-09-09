// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief This file contains the implementation for metadata traits.

#include "metadata_traits.h"

#include <viame/core_types/metadata.h>
#include <viame/algorithm_framework/util/demangle.h>

namespace kwiver {

namespace vital {

namespace {

// ----------------------------------------------------------------------------
std::vector< metadata_tag_traits > const&
tag_traits()
{
#define TRAITS_OF( X ) \
VITAL_META_##X, #X, typeid( type_of_tag< VITAL_META_##X > )
  static auto const array = std::vector< metadata_tag_traits >{
    { TRAITS_OF( UNKNOWN ),
      "Unknown / Undefined Entry",
      "Unknown or undefined entry." },
    { TRAITS_OF( METADATA_ORIGIN ),
      "Origin of Metadata",
      "Name of the metadata standard used to decode these metadata values "
      "from a video stream." },
    { TRAITS_OF( UNIX_TIMESTAMP ),
      "Unix Timestamp (microseconds)",
      "Number of microseconds since the Unix epoch, not counting leap "
      "seconds." },
    { TRAITS_OF( UNIX_NANO_TIMESTAMP ),
      "Unix Timestamp (nanoseconds)",
      "Number of nanoseconds since the Unix epoch, not counting leap "
      "seconds." },
    { TRAITS_OF( UNIX_TIMESTAMP_SOURCE ),
      "Unix Timestamp Source",
      "Detail about where the Unix timestamp was derived from. Currently "
      "permitted values: 'klv', 'misp'." },
    { TRAITS_OF( MISSION_ID ),
      "Mission ID",
      "Descriptive mission identifier to distinguish event or sortie." },
    { TRAITS_OF( MISSION_NUMBER ),
      "Episode Number",
      "Episode number." },
    { TRAITS_OF( PLATFORM_TAIL_NUMBER ),
      "Platform Tail Number",
      "Identifier of platform as posted." },
    { TRAITS_OF( PLATFORM_ORIENTATION ),
      "Platform Orientation",
      "Absolute orientation of platform using North-East-Down (NED) axes." },
    { TRAITS_OF( PLATFORM_TRUE_AIRSPEED ),
      "Platform True Airspeed (meters/second)",
      "True airspeed of platform. Indicated airspeed adjusted for temperature "
      "and altitude." },
    { TRAITS_OF( PLATFORM_INDICATED_AIRSPEED ),
      "Platform Indicated Airspeed (meters/second)",
      "Indicated airspeed of platform. Derived from Pitot tube and static "
      "pressure sensors." },
    { TRAITS_OF( PLATFORM_DESIGNATION ),
      "Platform Designation",
      "Platform designation." },
    { TRAITS_OF( IMAGE_SOURCE_SENSOR ),
      "Image Source Sensor",
      "String of image source sensor.  E.g.: 'EO Nose', 'EO Zoom (DLTV)', "
      "'EO Spotter', 'IR Mitsubishi PtSi Model 500', 'IR InSb Amber Model "
      "TBT', 'LYNX SAR Imagery', 'TESAR Imagery', etc." },
    { TRAITS_OF( IMAGE_COORDINATE_SYSTEM ),
      "Image Coordinate System",
      "Coordinate system used. E.g.: 'Geodetic WGS84', 'Geocentric WGS84', "
      "'TUM', 'None', etc." },
    { TRAITS_OF( IMAGE_URI ),
      "Image URI",
      "URI of source image." },
    { TRAITS_OF( IMAGE_WIDTH ),
      "Image Width",
      "Width of image in pixels." },
    { TRAITS_OF( IMAGE_HEIGHT ),
      "Image Height",
      "Height of image in pixels." },
    { TRAITS_OF( IMAGE_CORRUPT ),
      "Image Corrupt",
      "True if this image is wholly or partially corrupted." },
    { TRAITS_OF( VIDEO_DATA_STREAM_INDEX ),
      "Index of Metadata Stream",
      "Index of metadata stream." },
    { TRAITS_OF( VIDEO_DATA_STREAM_SYNCHRONOUS ),
      "Is Metadata Stream Synchronous",
      "True if the association between the metadata in this stream and the "
      "frame images is exact." },
    { TRAITS_OF( VIDEO_URI ),
      "Video URI",
      "URI of source video." },
    { TRAITS_OF( VIDEO_KEY_FRAME ),
      "Is Key Frame",
      "True if the current frame is a key frame." },
    { TRAITS_OF( VIDEO_FRAME_NUMBER ),
      "Frame Number",
      "Frame number of video input." },
    { TRAITS_OF( VIDEO_MICROSECONDS ),
      "Video Relative Timestamp",
      "Microseconds since beginning of video input." },
    { TRAITS_OF( VIDEO_FRAME_RATE ),
      "Frame Rate",
      "Average time in seconds between consecutive frames." },
    { TRAITS_OF( VIDEO_BITRATE ),
      "Video Bitrate",
      "Average bits per second of video." },
    { TRAITS_OF( VIDEO_COMPRESSION_TYPE ),
      "Video Compression Type",
      "Method of video compression used. Examples: 'H.262', 'H.264', "
      "'N/A' (uncompressed)." },
    { TRAITS_OF( VIDEO_COMPRESSION_PROFILE ),
      "Video Compression Profile",
      "Compression profile used to encode video. Examples: 'Main', "
      "'High 4:2:2', 'Main 4:2:2 12'." },
    { TRAITS_OF( VIDEO_COMPRESSION_LEVEL ),
      "Video Compression Level",
      "Compression level used to encode video. Exmaples: '4.1', 'ML'." },
    { TRAITS_OF( SENSOR_LOCATION ),
      "Sensor Geodetic Location (lon/lat/alt)",
      "Three-dimensional coordinates of the sensor: longitude, latitude, and "
      "(optionally) altitude." },
    { TRAITS_OF( SENSOR_HORIZONTAL_FOV ),
      "Sensor Horizonal Field of View (degrees)",
      "Horizontal field of view of selected imaging sensor." },
    { TRAITS_OF( SENSOR_VERTICAL_FOV ),
      "Sensor Vertical Field of View (degrees)",
      "Vertical field of view of selected imaging sensor." },
    { TRAITS_OF( SENSOR_ORIENTATION ),
      "Sensor Orientation",
      "Absolute orientation of sensor using North-East-Down (NED) axes." },
    { TRAITS_OF( SENSOR_TYPE ),
      "Sensor Type",
      "" },
    { TRAITS_OF( SLANT_RANGE ),
      "Slant Range (meters)",
      "Distance to target." },
    { TRAITS_OF( TARGET_WIDTH ),
      "Target Width (meters)",
      "Target width within sensor field of view." },
    { TRAITS_OF( FRAME_CENTER ),
      "Geodetic Frame Center (lon/lat/alt)",
      "Three-dimensional coordinates of the frame center: longitude, "
      "latitude, and (optionally) altitude." },
    { TRAITS_OF( CORNER_POINTS ),
      "Corner Points (lon/lat)",
      "Four sided polygon representing the image bounds. The corners are "
      "ordered: upper left, upper right, lower right, lower left." },
    { TRAITS_OF( WIND_DIRECTION ),
      "Wind Direction (degrees)",
      "Wind direction at aircraft location. This is the direction the wind is "
      "coming from relative to true north." },
    { TRAITS_OF( WIND_SPEED ),
      "Wind Speed (meters/second)",
      "Wind speed at aircraft location." },
    { TRAITS_OF( STATIC_PRESSURE ),
      "Static Pressure (millibar)",
      "Static pressure at aircraft location." },
    { TRAITS_OF( DENSITY_ALTITUDE ),
      "Density Altitude (meters)",
      "Density altitude at aircraft location. Relative aircraft performance "
      "metric based on outside air temperature, static pressure, and "
      "humidity." },
    { TRAITS_OF( OUTSIDE_AIR_TEMPERATURE ),
      "Outside Air Temperature (Celsius)",
      "Temperature outside aircraft." },
    { TRAITS_OF( TARGET_LOCATION ),
      "Target Geodetic Location (lon/lat/alt)",
      "Three-dimensional coordinates of the target: longitude, latitude, and "
      "(optionally) altitude." },
    { TRAITS_OF( TARGET_TRK_GATE_WIDTH ),
      "Target Track Gate Width (pixels)",
      "Tracking gate width (x value) of tracked target within field of view." },
    { TRAITS_OF( TARGET_TRK_GATE_HEIGHT ),
      "Target Track Gate Height (pixels)",
      "Tracking gate height (x value) of tracked target within field of view." },
    { TRAITS_OF( TARGET_ERROR_EST_CE90 ),
      "Target Error Estimate - CE90 (meters)",
      "Circular Error 90 (CE90) is the estimated error distance in the "
      "horizontal direction. Specifies the radius of 90% probability on a "
      "plane tangent to the earth’s surface." },
    { TRAITS_OF( TARGET_ERROR_EST_LE90 ),
      "Target Error Estimate - LE90 (meters)",
      "Lateral Error 90 (LE90) is the estimated error distance in the "
      "vertical (or lateral) direction. Specifies the interval of 90% "
      "probability in the local vertical direction." },
    { TRAITS_OF( DIFFERENTIAL_PRESSURE ),
      "Differential Pressure (millibar)",
      "Differential pressure at aircraft location. Measured as the stagnation/"
      "impact/total pressure minus static pressure." },
    { TRAITS_OF( PLATFORM_ANG_OF_ATTACK ),
      "Platform Angle of Attack (deg)",
      "Angle between platform longitudinal axis and relative wind. Positive "
      "angles for upward relative wind." },
    { TRAITS_OF( PLATFORM_VERTICAL_SPEED ),
      "Platform Vertical Speed (meters/sec)",
      "Vertical speed of the aircraft relative to zenith. Positive ascending, "
      "negative descending." },
    { TRAITS_OF( PLATFORM_SIDESLIP_ANGLE ),
      "Platform Sideslip Angle (degrees)",
      "The sideslip angle is the angle between the platform longitudinal axis "
      "and relative wind. Positive angles to right wing, negative to left." },
    { TRAITS_OF( AIRFIELD_BAROMET_PRESS ),
      "Airfield Barometric Pressure (millibars)",
      "Local pressure at airfield of known height." },
    { TRAITS_OF( AIRFIELD_ELEVATION ),
      "Airfield Elevation (meters)",
      "Elevation of airfield corresponding to Airfield Barometric Pressure." },
    { TRAITS_OF( RELATIVE_HUMIDITY ),
      "Relative Humidity (percent)",
      "Relative humidity at aircraft location." },
    { TRAITS_OF( PLATFORM_GROUND_SPEED ),
      "Platform Ground Speed (meters/second)",
      "Speed projected to the ground of an airborne platform passing "
      "overhead." },
    { TRAITS_OF( GROUND_RANGE ),
      "Ground Range (meters)",
      "Horizontal distance from ground position of aircraft relative to nadir,"
      " and target of interest." },
    { TRAITS_OF( PLATFORM_FUEL_REMAINING ),
      "Platform Fuel Remaining (kilograms)",
      "" },
    { TRAITS_OF( PLATFORM_CALL_SIGN ),
      "Platform Call Sign",
      "Call sign of platform or operating unit." },
    { TRAITS_OF( LASER_PRF_CODE ),
      "Laser PRF Code",
      "Pulse Repetition Frequency code used to mark a target. Three or four "
      "digit number consisting only of the digits 1..8." },
    { TRAITS_OF( SENSOR_FOV_NAME ),
      "Sensor Field of View Name",
      "Names sensor field of view level in quantized steps: 4x Ultranarrow, "
      "2x Ultranarrow, Ultranarrow, Narrow, Narrow Medium, Medium, Wide, "
      "Ultrawide." },
    { TRAITS_OF( PLATFORM_MAGNET_HEADING ),
      "Platform Magnetic Heading (degrees)",
      "Aircraft magnetic heading angle. Relative between longitudinal axis and"
      " Magnetic North measured in the horizontal plane." },
    { TRAITS_OF( UAS_LDS_VERSION_NUMBER ),
      "UAS LDS Version Number",
      "" },
    { TRAITS_OF( START_TIMESTAMP ),
      "Start Timestamp",
      "Time of collection start. Microseconds since UNIX epoch." },
    { TRAITS_OF( EVENT_START_TIMESTAMP ),
      "Event Start Timestamp",
      "Time of event, mission, etc. start. Microseconds since UNIX epoch." },
    { TRAITS_OF( SECURITY_CLASSIFICATION ),
      "Security Classification",
      "Security classification of source imagery." },
    { TRAITS_OF( AVERAGE_GSD ),
      "Average Ground Sample Distance (meters/pixel)",
      "" },
    { TRAITS_OF( VNIIRS ),
      "Video National Imagery Interpretability Rating Scale",
      "A subjective quality scale from 2 to 11 for rating the intelligence "
      "value of airborne motion imagery in the visible spectrum. See "
      "https://gwg.nga.mil/misb/docs/standards/ST0901.2.pdf" },
    { TRAITS_OF( WAVELENGTH ),
      "Sensor Wavelength",
      "Wavelength band currently in use. Standardized values: 'VIS' "
      "(visible), 'IR' (infrared), 'NIR' (near/short-wave infrared), 'MIR' "
      "(mid-wave infrared), 'LIR' (long-wave infrared), 'FIR' "
      "(far-infrared)." },
    { TRAITS_OF( GPS_SEC ),
      "GPS Seconds",
      "" },
    { TRAITS_OF( GPS_WEEK ),
      "GPS Week",
      "" },
    { TRAITS_OF( NORTHING_VEL ),
      "Northing Velocity (meters/second)",
      "" },
    { TRAITS_OF( EASTING_VEL ),
      "Easting Velocity (meters/second)",
      "" },
    { TRAITS_OF( UP_VEL ),
      "Up Velocity (meters/second)",
      "" },
    { TRAITS_OF( IMU_STATUS ),
      "IMU Status",
      "" },
    { TRAITS_OF( LOCAL_ADJ ),
      "Local Adj",
      "" },
    { TRAITS_OF( DST_FLAGS ),
      "Dst Flags",
      "" },
    { TRAITS_OF( RPC_HEIGHT_OFFSET ),
      "RPC Height Offset",
      "" },
    { TRAITS_OF( RPC_HEIGHT_SCALE ),
      "RPC Height Scale",
      "" },
    { TRAITS_OF( RPC_LONG_OFFSET ),
      "RPC Longitude Offset",
      "" },
    { TRAITS_OF( RPC_LONG_SCALE ),
      "RPC Longitude Scale",
      "" },
    { TRAITS_OF( RPC_LAT_OFFSET ),
      "RPC Latitude Offset",
      "" },
    { TRAITS_OF( RPC_LAT_SCALE ),
      "RPC Latitude Scale",
      "" },
    { TRAITS_OF( RPC_ROW_OFFSET ),
      "RPC Row Offset",
      "" },
    { TRAITS_OF( RPC_ROW_SCALE ),
      "RPC Row Scale",
      "" },
    { TRAITS_OF( RPC_COL_OFFSET ),
      "RPC Column Offset",
      "" },
    { TRAITS_OF( RPC_COL_SCALE ),
      "RPC Column Scale",
      "" },
    { TRAITS_OF( RPC_ROW_NUM_COEFF ),
      "RPC Row Numerator Coefficients",
      "" },
    { TRAITS_OF( RPC_ROW_DEN_COEFF ),
      "RPC Row Denominator Coefficients",
      "" },
    { TRAITS_OF( RPC_COL_NUM_COEFF ),
      "RPC Column Numerator Coefficients",
      "" },
    { TRAITS_OF( RPC_COL_DEN_COEFF ),
      "RPC Column Denominator Coefficients",
      "" },
    { TRAITS_OF( NITF_IDATIM ),
      "NITF IDATIM",
      "" },
    { TRAITS_OF( NITF_BLOCKA_FRFC_LOC_01 ),
      "First Row First Column Location",
      "" },
    { TRAITS_OF( NITF_BLOCKA_FRLC_LOC_01 ),
      "First Row Last Column Location",
      "" },
    { TRAITS_OF( NITF_BLOCKA_LRFC_LOC_01 ),
      "Last Row First Column Location",
      "" },
    { TRAITS_OF( NITF_BLOCKA_LRLC_LOC_01 ),
      "Last Row Last Column Location",
      "" },
    { TRAITS_OF( NITF_IMAGE_COMMENTS ),
      "Image Comments for NITF File",
      "" }, };
  return array;
}

// ----------------------------------------------------------------------------
metadata_tag_traits const&
unknown_tag_traits()
{
  return tag_traits().at( 0 );
}

} // namespace

// ----------------------------------------------------------------------------
metadata_tag_traits
::metadata_tag_traits(
  vital_metadata_tag tag, std::string const& enum_name,
  std::type_info const& type, std::string const& name,
  std::string const& description )
  : m_tag{ tag },
    m_enum_name{ enum_name },
    m_type( type ),
    m_name{ name },
    m_description{ description }
{}

// ----------------------------------------------------------------------------
vital_metadata_tag
metadata_tag_traits
::tag() const
{
  return m_tag;
}

// ----------------------------------------------------------------------------
std::string
metadata_tag_traits
::name() const
{
  return m_name;
}

// ----------------------------------------------------------------------------
std::string
metadata_tag_traits
::enum_name() const
{
  return m_enum_name;
}

// ----------------------------------------------------------------------------
std::type_info const&
metadata_tag_traits
::type() const
{
  return m_type;
}

// ----------------------------------------------------------------------------
std::string
metadata_tag_traits
::type_name() const
{
  static std::map< std::type_index, std::string > const map = {
    { typeid( bool ), "bool" },
    { typeid( int ), "int" },
    { typeid( uint64_t ), "uint64" },
    { typeid( double ), "double" },
    { typeid( std::string ), "string" },
    { typeid( geo_point ), "geo_point" },
    { typeid( geo_polygon ), "geo_polygon" },
    { typeid( rotation_d ), "rotation_d" }, };

  return map.at( m_type );
}

// ----------------------------------------------------------------------------
std::string
metadata_tag_traits
::description() const
{
  return m_description;
}

// ----------------------------------------------------------------------------
metadata_tag_traits const&
tag_traits_by_tag( vital_metadata_tag tag )
{
  static auto const map =
    ( [](){
        std::map< vital_metadata_tag, metadata_tag_traits const* > result;
        for( auto const& traits : tag_traits() )
        {
          result.emplace( traits.tag(), &traits );
        }
        return result;
      } )();
  auto const it = map.find( tag );
  return ( it == map.end() ) ? unknown_tag_traits() : *it->second;
}

// ----------------------------------------------------------------------------
metadata_tag_traits const&
tag_traits_by_name( std::string const& name )
{
  static auto const map =
    ( [](){
        std::map< std::string, metadata_tag_traits const* > result;
        for( auto const& traits : tag_traits() )
        {
          result.emplace( traits.name(), &traits );
        }
        return result;
      } )();
  auto const it = map.find( name );
  return ( it == map.end() ) ? unknown_tag_traits() : *it->second;
}

// ----------------------------------------------------------------------------
metadata_tag_traits const&
tag_traits_by_enum_name( std::string const& name )
{
  static auto const map =
    ( [](){
        std::map< std::string, metadata_tag_traits const* > result;
        for( auto const& traits : tag_traits() )
        {
          result.emplace( traits.enum_name(), &traits );
        }
        return result;
      } )();
  auto const it = map.find( name );
  return ( it == map.end() ) ? unknown_tag_traits() : *it->second;
}

} // namespace vital

} // namespace kwiver
