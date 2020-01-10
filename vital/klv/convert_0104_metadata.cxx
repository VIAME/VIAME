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
 * \brief This file contains the implementation for vital video metadata.
 */

#include "convert_metadata.h"

#include <vital/klv/klv_0104.h>
#include <vital/klv/klv_data.h>

#include <vital/exceptions/metadata.h>

#include <vital/types/geodesy.h>

namespace kwiver {
namespace vital {

namespace {

// ----------------------------------------------------------------------------
template < int N >
Eigen::Matrix< double, N, 1 >
empty_vector()
{
  Eigen::Matrix< double, N, 1 > v;
  for ( int i = 0; i < N; ++i )
  {
    v[i] = std::numeric_limits<double>::quiet_NaN();
  }

  return v;
}

// ----------------------------------------------------------------------------
template < int N >
bool
is_empty( Eigen::Matrix< double, N, 1, 0, N, 1 > const& vec )
{
  for ( int i = 0; i < N; ++i )
  {
    if ( std::isnan( vec[i] ) )
    {
      return true;
    }
  }

  return false;
}

// ----------------------------------------------------------------------------
bool
is_valid_lon_lat( vector_2d const& vec )
{
  auto const lat = vec[1];
  auto const lon = vec[0];
  return ( lat >= -90.0 && lat <= 90.0 ) &&
         ( lon >= -180.0 && lon <= 360.0 );
}

// ----------------------------------------------------------------------------
bool
is_valid_lon_lat( vector_3d const& vec )
{
  return is_valid_lon_lat( static_cast< vector_2d >(vec.head(2)) );
}

} // end namespace

// ----------------------------------------------------------------------------
/**
 * @brief Normalize metadata tag data types.
 *
 * @param[in] vital_tag Metadata tag
 * @param[in,out] data Data to be normalized
 */
kwiver::vital::any
convert_metadata
::normalize_0104_tag_data( klv_0104::tag tag,
                           kwiver::vital::vital_metadata_tag vital_tag,
                           kwiver::vital::any const& data )
{
  // If the input data is already in the correct type, just return.
  if ( metadata::typeid_for_tag( vital_tag ) == data.type() )
  {
    return data;
  }

  try
  {
    // If destination type is double, then source must be convertable to double
    if ( metadata::typeid_for_tag( vital_tag ) == typeid( double ) )
    {
      kwiver::vital::any converted_data = convert_to_double.convert( data );
      return converted_data;
    }

    // If the destination is integral.
    vital_meta_trait_base const& trait = m_metadata_traits.find( vital_tag );
    if ( trait.is_integral() )
    {
      kwiver::vital::any converted_data = convert_to_int.convert( data );
      return converted_data;
    }
  }
  catch (kwiver::vital::bad_any_cast const& e)
  {
    LOG_DEBUG( m_logger, "Data not convertable for tag: "
              << m_metadata_traits.tag_to_symbol( vital_tag )
              << ",  " << e.what() );
  }

  return data;
}


// ------------------------------------------------------------------
void convert_metadata
::convert_0104_metadata( klv_uds_vector_t const& uds, metadata& md )
{
  //
  // Data items that are used to collect multi-value metadata items such as
  // lat-lon points and image corner points. All geodetic points are assumed to
  // be WGS84 lat-lon.
  //
  auto raw_sensor_location = empty_vector<3>();
  auto raw_frame_center = empty_vector<2>();
  auto raw_corner_pt1 = empty_vector<2>(); // offsets relative to frame_center
  auto raw_corner_pt2 = empty_vector<2>();
  auto raw_corner_pt3 = empty_vector<2>();
  auto raw_corner_pt4 = empty_vector<2>();

  //
  // Add our "origin" tag to indicate that the source of this metadata
  // collection is from a 0104 spec packet.
  //
  md.add( NEW_METADATA_ITEM( VITAL_META_METADATA_ORIGIN, MISB_0104 ) );

  for ( auto itr = uds.begin(); itr != uds.end(); ++itr )
  {
    klv_0104::tag tag;
    kwiver::vital::any data;

    try
    {
      tag = klv_0104::instance()->get_tag( itr->first );
      if ( tag == klv_0104::UNKNOWN )
      {
        LOG_DEBUG( m_logger, "Unknown key: " << itr->first << "Length: " << itr->second.size() << " bytes" );
        continue;
      }

      data = klv_0104::instance()->get_value( tag, &itr->second[0], itr->second.size() );
    }
    catch ( kwiver::vital::metadata_exception const& e )
    {
      LOG_INFO( m_logger, "Exception caught parsing 0104 klv: " << e.what() );
      continue;
    }

    //
    // Data items that are used to collect multi-value metadata items
    // such as lat-lon points and image corner points.
    //

    switch (tag)
    {
// Refine simple case to a define
#define CASE(N)                                           \
case klv_0104::N:                                         \
  md.add( NEW_METADATA_ITEM( VITAL_META_ ## N, data ) );  \
  break

#define CASE2(KN,MN)                                         \
      case klv_0104::KN:                                     \
    md.add( NEW_METADATA_ITEM( VITAL_META_ ## MN, data ) );  \
    break

      CASE( UNIX_TIMESTAMP );
      CASE( MISSION_ID );
      CASE( MISSION_NUMBER );
      CASE( PLATFORM_TAIL_NUMBER );
      CASE( PLATFORM_HEADING_ANGLE );
      CASE( PLATFORM_PITCH_ANGLE );
      CASE( PLATFORM_ROLL_ANGLE );
      CASE( PLATFORM_TRUE_AIRSPEED );
      CASE( PLATFORM_INDICATED_AIRSPEED );
      CASE( PLATFORM_DESIGNATION );
      CASE( IMAGE_SOURCE_SENSOR );
      CASE( IMAGE_COORDINATE_SYSTEM );
      CASE( SENSOR_HORIZONTAL_FOV );
      CASE( SENSOR_VERTICAL_FOV );
      CASE( SENSOR_ROLL_ANGLE );
      CASE2( SENSOR_RELATIVE_ROLL_ANGLE, SENSOR_REL_ROLL_ANGLE );
      CASE( SLANT_RANGE );
      CASE( TARGET_WIDTH );
      CASE( WIND_DIRECTION );
      CASE( WIND_SPEED );
      CASE( PLATFORM_CALL_SIGN );
      CASE2( FOV_NAME, SENSOR_FOV_NAME );
      CASE( ANGLE_TO_NORTH );
      CASE( OBLIQUITY_ANGLE );
      CASE( START_DATE_TIME_UTC );
      CASE2( MISSION_START_TIME, MISSION_START_TIME_UTC );
      CASE( SECURITY_CLASSIFICATION );
      CASE( CLASSIFICATION );
      CASE( SENSOR_TYPE );
      CASE( EVENT_START_DATE_TIME_UTC );

#undef CASE
#undef CASE2

    case klv_0104::SENSOR_ALTITUDE:
      raw_sensor_location[2] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::SENSOR_LATITUDE:
      raw_sensor_location[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::SENSOR_LONGITUDE:
      raw_sensor_location[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::FRAME_CENTER_LATITUDE:
      raw_frame_center[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::FRAME_CENTER_LONGITUDE:
      raw_frame_center[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::UPPER_LEFT_CORNER_LAT:
      raw_corner_pt1[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::UPPER_LEFT_CORNER_LON:
      raw_corner_pt1[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::UPPER_RIGHT_CORNER_LAT:
      raw_corner_pt2[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::UPPER_RIGHT_CORNER_LON:
      raw_corner_pt2[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::LOWER_RIGHT_CORNER_LAT:
      raw_corner_pt3[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::LOWER_RIGHT_CORNER_LON:
      raw_corner_pt3[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::LOWER_LEFT_CORNER_LAT:
      raw_corner_pt4[1] =  kwiver::vital::any_cast< double >(data) ;
      break;

    case klv_0104::LOWER_LEFT_CORNER_LON:
      raw_corner_pt4[0] =  kwiver::vital::any_cast< double >(data) ;
      break;

    default:
      LOG_DEBUG( m_logger, "Unprocessed key: " << itr->first << "Length: " << itr->second.size() << " bytes" );
      break;
    } // end switch

  } // end for

  //
  // Process composite metadata
  //
  if ( ! is_empty( raw_sensor_location ) )
  {
    if ( ! is_valid_lon_lat( raw_sensor_location ) )
    {
      LOG_DEBUG( m_logger, "Sensor location lat/lon is not valid coordinate: " << raw_sensor_location );
    }
    else
    {
      vector_3d sensor_loc(raw_sensor_location[0], raw_sensor_location[1], raw_sensor_location[2]);
      auto const sensor_location = geo_point{ sensor_loc, SRID::lat_lon_WGS84 };
      md.add( NEW_METADATA_ITEM( VITAL_META_SENSOR_LOCATION, sensor_location ) );
    }
  }

  if ( ! is_empty( raw_frame_center ) )
  {
    if ( ! is_valid_lon_lat( raw_frame_center ) )
    {
      LOG_DEBUG( m_logger, "Frame Center lat/lon is not valid coordinate: " << raw_frame_center );
    }
    else
    {
      auto const frame_center = geo_point{ raw_frame_center, SRID::lat_lon_WGS84 };
      md.add( NEW_METADATA_ITEM( VITAL_META_FRAME_CENTER, frame_center ) );
    }
  }

  //
  // If none of the points are set, then that is o.k.
  //
  if ( ! ( is_empty( raw_corner_pt1 ) &&
           is_empty( raw_corner_pt2 ) &&
           is_empty( raw_corner_pt3 ) &&
           is_empty( raw_corner_pt4 ) ) )
  {
    // If any one of the points are invalid, then decode which one
    if ( ! is_valid_lon_lat( raw_corner_pt1 ) ||
         ! is_valid_lon_lat( raw_corner_pt2 ) ||
         ! is_valid_lon_lat( raw_corner_pt3 ) ||
         ! is_valid_lon_lat( raw_corner_pt4 ) )
    {
      // Decode which one(s) are not valid
      if ( ! is_valid_lon_lat( raw_corner_pt1 ) )
      {
        LOG_DEBUG( m_logger, "Corner point 1 lat/lon is not valid coordinate: " << raw_corner_pt1 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt2 ) )
      {
        LOG_DEBUG( m_logger, "Corner point 2 lat/lon is not valid coordinate: " << raw_corner_pt2 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt3 ) )
      {
        LOG_DEBUG( m_logger, "Corner point 3 lat/lon is not valid coordinate: " << raw_corner_pt3 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt4 ) )
      {
        LOG_DEBUG( m_logger, "Corner point 4 lat/lon is not valid coordinate: " << raw_corner_pt4 );
      }
    }
    else
    {
      // If all points are set and valid, then build corner point structure
      kwiver::vital::polygon raw_corners;

      raw_corners.push_back( raw_corner_pt1 );
      raw_corners.push_back( raw_corner_pt2 );
      raw_corners.push_back( raw_corner_pt3 );
      raw_corners.push_back( raw_corner_pt4 );

      kwiver::vital::geo_polygon corners{ raw_corners, kwiver::vital::SRID::lat_lon_WGS84 };
      md.add( NEW_METADATA_ITEM( VITAL_META_CORNER_POINTS, corners ) );
    }
  }
}

} } // end namespace
