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

#include <vital/types/metadata.h>
#include "convert_metadata.h"

#include <vital/klv/klv_0601.h>
#include <vital/klv/klv_0601_traits.h>
#include <vital/klv/klv_data.h>

#include <vital/logger/logger.h>

#include <vital/types/geodesy.h>

#include <memory>
#include <type_traits>

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
  return is_valid_lon_lat( static_cast< vector_2d >(vec.head( 2 )) );
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
::normalize_0601_tag_data( klv_0601_tag tag,
                           kwiver::vital::vital_metadata_tag vital_tag,
                           kwiver::vital::any const& data )
{
  LOG_TRACE( m_logger, "Converting 0601 tag to Vital: "
             << m_metadata_traits.tag_to_symbol( vital_tag ) );

  // If the input data is already in the correct type
  if ( metadata::typeid_for_tag( vital_tag ) == data.type() )
  {
    // leave data as is since it already correct type.
    return data;
  }


  // If destination type is double, then source must be convertable to double
  if ( metadata::typeid_for_tag( vital_tag ) == typeid( double ) )
  {
    if ( klv_0601_has_double( tag ) )
    {
      kwiver::vital::any converted_data = klv_0601_value_double( tag, data );
      return converted_data;
    }
    // Could use convert_to_double.convert here to coerce the type if
    // we believe the tags tables are correct.
  }

  try
  {
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

    LOG_DEBUG( m_logger, "Tag data not converted for tag: "
              << m_metadata_traits.tag_to_symbol( vital_tag ) );

  return data;
}


// ------------------------------------------------------------------
void
convert_metadata
::convert_0601_metadata( klv_lds_vector_t const& lds, metadata& md )
{
  static kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.convert_metadata" ) );

  md.add( NEW_METADATA_ITEM( VITAL_META_METADATA_ORIGIN, MISB_0601 ) );

  //
  // Data items that are used to collect multi-value metadataa items such as
  // lat-lon points and image corner points. All geodetic points are assumed to
  // be WGS84 lat-lon.
  //
  auto raw_sensor_location = empty_vector<3>();
  auto raw_frame_center = empty_vector<3>();
  auto raw_corner_pt1 = empty_vector<2>(); // offsets relative to frame_center
  auto raw_corner_pt2 = empty_vector<2>();
  auto raw_corner_pt3 = empty_vector<2>();
  auto raw_corner_pt4 = empty_vector<2>();
  auto raw_target_location = empty_vector<3>();

  for ( auto itr = lds.begin(); itr != lds.end(); ++itr )
  {
    if ( ( itr->first <= KLV_0601_UNKNOWN ) || ( itr->first >= KLV_0601_ENUM_END ) )
    {
      LOG_DEBUG( logger, "KLV 0601 key: " << int(itr->first) << " is not supported" );
      continue;
    }

    // Convert a single tag
    const klv_0601_tag tag( klv_0601_get_tag( itr->first ) ); // get tag code from key

    LOG_TRACE( logger, "Processing 0601 tag: "
               << klv_0601_tag_to_string( tag ) );


    // Extract relevant data from associated data bytes.
    kwiver::vital::any data = klv_0601_value( tag,
                                              &itr->second[0], itr->second.size() );
    switch (tag)
    {
// Refine simple case to a define
#define CASE(N)                                                         \
  case KLV_0601_ ## N:                                                  \
    md.add( NEW_METADATA_ITEM( VITAL_META_ ## N,                        \
      normalize_0601_tag_data( KLV_0601_ ## N, VITAL_META_ ## N, data ) ) ); \
    break

#define CASE_COPY(N)                                                    \
  case KLV_0601_ ## N:                                                  \
    md.add( NEW_METADATA_ITEM( VITAL_META_ ## N, data ) );              \
    break

#define CASE2(KN,VN)                                                    \
  case KLV_0601_ ## KN:                                                 \
    md.add( NEW_METADATA_ITEM( VITAL_META_ ## VN,                       \
      normalize_0601_tag_data( KLV_0601_ ## KN, VITAL_META_ ## VN, data ) ) ); \
    break

      CASE( UNIX_TIMESTAMP );
      CASE( MISSION_ID );
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
      CASE( SENSOR_REL_AZ_ANGLE );
      CASE( SENSOR_REL_EL_ANGLE );
      CASE( SENSOR_REL_ROLL_ANGLE );
      CASE( SLANT_RANGE );
      CASE( TARGET_WIDTH );
      CASE( ICING_DETECTED);
      CASE( WIND_DIRECTION );
      CASE( WIND_SPEED );
      CASE( STATIC_PRESSURE );
      CASE( DENSITY_ALTITUDE );
      CASE( OUTSIDE_AIR_TEMPERATURE );
      CASE( TARGET_TRK_GATE_WIDTH );
      CASE( TARGET_TRK_GATE_HEIGHT );
      CASE_COPY( SECURITY_LOCAL_MD_SET );
      CASE( TARGET_ERROR_EST_CE90 );
      CASE( TARGET_ERROR_EST_LE90 );
      CASE( DIFFERENTIAL_PRESSURE );
      CASE( PLATFORM_ANG_OF_ATTACK );
      CASE( PLATFORM_VERTICAL_SPEED );
      CASE( PLATFORM_SIDESLIP_ANGLE );
      CASE( AIRFIELD_BAROMET_PRESS );
      CASE( AIRFIELD_ELEVATION );
      CASE( RELATIVE_HUMIDITY );
      CASE( PLATFORM_GROUND_SPEED );
      CASE( GROUND_RANGE );
      CASE( PLATFORM_FUEL_REMAINING );
      CASE( PLATFORM_CALL_SIGN );
      CASE( LASER_PRF_CODE );
      CASE( SENSOR_FOV_NAME );
      CASE( PLATFORM_MAGNET_HEADING );
      CASE( UAS_LDS_VERSION_NUMBER );

      // Source specific metadata tags

      // These are prefixed with the spec. number because the data format is specification specific.
      CASE2( WEAPON_LOAD, 0601_WEAPON_LOAD );
      CASE2( WEAPON_FIRED, 0601_WEAPON_FIRED );

#undef CASE
#undef CASE2

      // option (2) Use klv 0601 native to double converter.

    case KLV_0601_SENSOR_TRUE_ALTITUDE:
      raw_sensor_location[2] = klv_0601_value_double( KLV_0601_SENSOR_TRUE_ALTITUDE, data );
      break;

    case KLV_0601_SENSOR_LATITUDE:
      raw_sensor_location[1] = klv_0601_value_double( KLV_0601_SENSOR_LATITUDE, data );
      break;

    case KLV_0601_SENSOR_LONGITUDE:
      raw_sensor_location[0] = klv_0601_value_double( KLV_0601_SENSOR_LONGITUDE, data );
      break;

    case KLV_0601_FRAME_CENTER_LAT:
      raw_frame_center[1] = klv_0601_value_double( KLV_0601_FRAME_CENTER_LAT, data );
      break;

    case KLV_0601_FRAME_CENTER_LONG:
      raw_frame_center[0] = klv_0601_value_double( KLV_0601_FRAME_CENTER_LONG, data );
      break;

    case KLV_0601_FRAME_CENTER_ELEV:
      raw_frame_center[2] = klv_0601_value_double( KLV_0601_FRAME_CENTER_ELEV, data );
      break;

      // Sometimes these offsets are set to zero. Even if the image is
      // really that small, we can not create a meaningfull bounding
      // box.  Currently we are ignoring the metadata if the offsets
      // are zero. One could argue that the bounding box should be
      // created and application level semantics should decide if it
      // is meaningful or not.
    case KLV_0601_OFFSET_CORNER_LAT_PT_1:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LAT_PT_1, data );
      if (temp != 0)
      {
        raw_corner_pt1[1] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LONG_PT_1:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LONG_PT_1, data );
      if (temp != 0)
      {
        raw_corner_pt1[0] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LAT_PT_2:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LAT_PT_2, data );
      if (temp != 0)
      {
        raw_corner_pt2[1] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LONG_PT_2:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LONG_PT_2, data );
      if (temp != 0)
      {
        raw_corner_pt2[0] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LAT_PT_3:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LAT_PT_3, data );
      if (temp != 0)
      {
        raw_corner_pt3[1] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LONG_PT_3:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LONG_PT_3, data );
      if (temp != 0)
      {
        raw_corner_pt3[0] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LAT_PT_4:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LAT_PT_4, data );
      if (temp != 0)
      {
        raw_corner_pt4[1] = temp;
      }
    }
      break;

    case KLV_0601_OFFSET_CORNER_LONG_PT_4:
    {
      double temp = klv_0601_value_double( KLV_0601_OFFSET_CORNER_LONG_PT_4, data );
      if (temp != 0)
      {
        raw_corner_pt4[0] = temp;
      }
    }
      break;

    case KLV_0601_TARGET_LOCATION_ELEV:
      raw_target_location[2] = klv_0601_value_double( KLV_0601_TARGET_LOCATION_LAT, data );
      break;

    case KLV_0601_TARGET_LOCATION_LAT:
      raw_target_location[1] = klv_0601_value_double( KLV_0601_TARGET_LOCATION_LAT, data );
      break;

    case KLV_0601_TARGET_LOCATION_LONG:
      raw_target_location[0] = klv_0601_value_double( KLV_0601_TARGET_LOCATION_LONG, data );
      break;

    default:
      LOG_DEBUG( logger, "KLV 0601 key: " << int(itr->first) << " is not supported." );
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
      LOG_DEBUG( logger, "Sensor location lat/lon is not valid coordinate: " << raw_sensor_location );
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
      LOG_DEBUG( logger, "Frame Center lat/lon is not valid coordinate: " << raw_frame_center );
    }
    else
    {
      auto const frame_center = geo_point{ raw_frame_center, SRID::lat_lon_WGS84 };
      md.add( NEW_METADATA_ITEM( VITAL_META_FRAME_CENTER, frame_center ) );
    }
  }

  if ( ! is_empty( raw_target_location ) )
  {
    if ( ! is_valid_lon_lat( raw_target_location ) )
    {
      LOG_DEBUG( logger, "Target location lat/lon is not valid coordinate: " << raw_target_location );
    }
    else
    {
      auto const target_location = geo_point{ raw_target_location, SRID::lat_lon_WGS84 };
      md.add( NEW_METADATA_ITEM( VITAL_META_TARGET_LOCATION, target_location ) );
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
        LOG_DEBUG( logger, "Corner point 1 lat/lon is not valid coordinate: " << raw_corner_pt1 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt2 ) )
      {
        LOG_DEBUG( logger, "Corner point 2 lat/lon is not valid coordinate: " << raw_corner_pt2 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt3 ) )
      {
        LOG_DEBUG( logger, "Corner point 3 lat/lon is not valid coordinate: " << raw_corner_pt3 );
      }

      if ( ! is_valid_lon_lat( raw_corner_pt4 ) )
      {
        LOG_DEBUG( logger, "Corner point 4 lat/lon is not valid coordinate: " << raw_corner_pt4 );
      }
    }
    else
    {
      if ( ! is_valid_lon_lat( raw_frame_center ) )
      {
        LOG_DEBUG( logger, "Frame center not valid. Can not adjust frame corner offsets." );
      }
      else
      {
        // If all points are set and valid, then build corner point structure
        kwiver::vital::polygon raw_corners;
        vector_2d rfc( raw_frame_center[0], raw_frame_center[1] );
        raw_corners.push_back( raw_corner_pt1 + rfc );
        raw_corners.push_back( raw_corner_pt2 + rfc );
        raw_corners.push_back( raw_corner_pt3 + rfc );
        raw_corners.push_back( raw_corner_pt4 + rfc );

        kwiver::vital::geo_polygon corners{ raw_corners, kwiver::vital::SRID::lat_lon_WGS84 };
        md.add( NEW_METADATA_ITEM( VITAL_META_CORNER_POINTS, corners ) );
      }
    }
  } // corner points are empty
}

} } // end namespace
