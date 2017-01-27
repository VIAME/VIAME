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
 * \brief This file contains the implementation for representing
 * and accessing klv 0104 video metadata.
 */

#include "klv_0104.h"
#include "klv_data.h"

#include <vital/exceptions/klv.h>
#include <vital/logger/logger.h>

#include <type_traits>
#include <sstream>
#include <map>
#include <iomanip>
#include <mutex>

namespace kwiver {
namespace vital {


class std_0102_lds { };


// ------------------------------------------------------------------
/// Provides interpretation of raw data to kwiver::vital::any and can also
/// convert the any to a string
template < class T >
class traits :
    public klv_0104::traits_base
{
public:
  traits( std::string const& name )
    : traits_base( name )
  { }

  traits( std::string const& name, bool set )
    : traits_base( name, set )
  { }

  virtual ~traits() { }

  virtual std::string to_string( kwiver::vital::any const& data ) const;

  /// Parse type T from a raw byte stream in MSB (most significant byte first) order
  virtual kwiver::vital::any convert( uint8_t const* data, std::size_t length );
  virtual std::type_info const& typeid_for_tag( ) const { return typeid(T); }
  virtual bool is_integral() const { return std::is_integral<T>::value; }
  virtual bool is_floating_point() const { return std::is_floating_point<T>::value; }

};


namespace {

// ==================================================================
static const uint8_t key_data[16] =
{
  0x06, 0x0E, 0x2B, 0x34,
  0x02, 0x01, 0x01, 0x01,
  0x0E, 0x01, 0x01, 0x02,
  0x01, 0x01, 0x00, 0x00
};

static const klv_uds_key klv_0104_uds_key( key_data );

}  //end anonymous namespace


klv_0104* klv_0104::s_instance = 0;  // instance pointer


// ------------------------------------------------------------------
klv_0104*
klv_0104::instance()
{
  static std::mutex local_lock;          // synchronization lock

  if (0 != s_instance)
  {
    return s_instance;
  }

  std::lock_guard<std::mutex> lock(local_lock);
  if (0 == s_instance)
  {
    // create new object
    s_instance = new klv_0104;
  }

  return s_instance;
}


// ------------------------------------------------------------------
klv_0104::klv_0104()
{
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0101200100000000UL )] = PLATFORM_DESIGNATION;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0101210100000000UL )] = PLATFORM_DESIGNATION_ALT;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0103040200000000UL )] = STREAM_ID;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0103060100000000UL )] = ITEM_DESIGNATOR_ID;
  m_key_to_tag[klv_uds_key( 0x060e2b3402010101UL, 0x0208020000000000UL )] = CLASSIFICATION;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0208020100000000UL )] = SECURITY_CLASSIFICATION;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0420010201010000UL )] = IMAGE_SOURCE_SENSOR;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010102UL, 0x0420020101080000UL )] = SENSOR_HORIZONTAL_FOV;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010107UL, 0x04200201010a0100UL )] = SENSOR_VERTICAL_FOV;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0420030100000000UL )] = SENSOR_TYPE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701010100000000UL )] = IMAGE_COORDINATE_SYSTEM;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701090201000000UL )] = TARGET_WIDTH;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010107UL, 0x0701100106000000UL )] = PLATFORM_HEADING_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010107UL, 0x0701100105000000UL )] = PLATFORM_PITCH_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010107UL, 0x0701100104000000UL )] = PLATFORM_ROLL_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0701020102040200UL )] = SENSOR_LATITUDE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0701020102060200UL )] = SENSOR_LONGITUDE;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0701020102020000UL )] = SENSOR_ALTITUDE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701020103020000UL )] = FRAME_CENTER_LATITUDE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701020103040000UL )] = FRAME_CENTER_LONGITUDE;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x0701020103070100UL )] = UPPER_LEFT_CORNER_LAT;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x07010201030B0100UL )] = UPPER_LEFT_CORNER_LON;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x0701020103080100UL )] = UPPER_RIGHT_CORNER_LAT;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x07010201030C0100UL )] = UPPER_RIGHT_CORNER_LON;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x0701020103090100UL )] = LOWER_RIGHT_CORNER_LAT;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x07010201030D0100UL )] = LOWER_RIGHT_CORNER_LON;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x07010201030A0100UL )] = LOWER_LEFT_CORNER_LAT;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010103UL, 0x07010201030E0100UL )] = LOWER_LEFT_CORNER_LON;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701080101000000UL )] = SLANT_RANGE;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0701100102000000UL )] = ANGLE_TO_NORTH;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0701100103000000UL )] = OBLIQUITY_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0702010201010000UL )] = START_DATE_TIME_UTC;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0702010207010000UL )] = EVENT_START_DATE_TIME_UTC;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010103UL, 0x0702010101050000UL )] = UNIX_TIMESTAMP;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e0101010a000000UL )] = PLATFORM_TRUE_AIRSPEED;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e0101010b000000UL )] = PLATFORM_INDICATED_AIRSPEED;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e01040101000000UL )] = PLATFORM_CALL_SIGN;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e01020202000000UL )] = FOV_NAME;
  m_key_to_tag[klv_uds_key( 0x060E2B3402010101UL, 0x0e0101010D000000UL )] = WIND_DIRECTION;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0e0101010E000000UL )] = WIND_SPEED;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0e01010201010000UL )] = PREDATOR_UAV_UMS;
  m_key_to_tag[klv_uds_key( 0x060E2B3402010101UL, 0x0e01010201010000UL )] = PREDATOR_UAV_UMS_V2;
  m_key_to_tag[klv_uds_key( 0x060E2B3401010101UL, 0x0e01010206000000UL )] = SENSOR_RELATIVE_ROLL_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e01040103000000UL )] = MISSION_ID;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0e01040102000000UL )] = PLATFORM_TAIL_NUMBER;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0105050000000000UL )] = MISSION_NUMBER;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701100101000000UL )] = SENSOR_ROLL_ANGLE;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701100102000000UL )] = ANGLE_TO_NORTH;
  m_key_to_tag[klv_uds_key( 0x060e2b3401010101UL, 0x0701100103000000UL )] = OBLIQUITY_ANGLE;

  //UNKNOWN is the last entry of the enum and thus the number of tags
  m_traitsvec.resize( UNKNOWN );

  //Mapping between tag index and traits, double is used for floats and doubles
#define NEW_TRAIT(T,D) static_cast< traits_base* >(new traits< T > ( D ))

  m_traitsvec[PLATFORM_DESIGNATION] =        NEW_TRAIT( std::string,  "Platform designation" );
  m_traitsvec[PLATFORM_DESIGNATION_ALT] =    NEW_TRAIT( std::string,  "Platform designation (alternate key)" );
  m_traitsvec[STREAM_ID] =                   NEW_TRAIT( std::string,  "Stream ID" );
  m_traitsvec[ITEM_DESIGNATOR_ID] =          NEW_TRAIT( std::string,  "Item Designator ID (16 bytes)" );
  m_traitsvec[CLASSIFICATION] =              NEW_TRAIT( std_0102_lds,  "Classification" );
  m_traitsvec[SECURITY_CLASSIFICATION] =     NEW_TRAIT( std::string,  "Security Classification" );
  m_traitsvec[IMAGE_SOURCE_SENSOR] =         NEW_TRAIT( std::string,  "Image Source sensor" );
  m_traitsvec[SENSOR_HORIZONTAL_FOV] =       NEW_TRAIT( double,       "Sensor horizontal field of view" );
  m_traitsvec[SENSOR_VERTICAL_FOV] =         NEW_TRAIT( double,       "Sensor vertical field of view" );
  m_traitsvec[SENSOR_TYPE] =                 NEW_TRAIT( std::string,  "Sensor type" );
  m_traitsvec[IMAGE_COORDINATE_SYSTEM] =     NEW_TRAIT( std::string,  "Image Coordinate System" );
  m_traitsvec[TARGET_WIDTH] =                NEW_TRAIT( double,       "Target Width" );
  m_traitsvec[PLATFORM_HEADING_ANGLE] =      NEW_TRAIT( double,       "Platform heading" );
  m_traitsvec[PLATFORM_PITCH_ANGLE] =        NEW_TRAIT( double,       "Platform pitch angle" );
  m_traitsvec[PLATFORM_ROLL_ANGLE] =         NEW_TRAIT( double,       "Platform roll angle" );
  m_traitsvec[SENSOR_LATITUDE] =             NEW_TRAIT( double,       "Sensor latitude" );
  m_traitsvec[SENSOR_LONGITUDE] =            NEW_TRAIT( double,       "Sensor longitude" );
  m_traitsvec[SENSOR_ALTITUDE] =             NEW_TRAIT( double,       "Sensor Altitude" );
  m_traitsvec[FRAME_CENTER_LATITUDE] =       NEW_TRAIT( double,       "Frame center latitude" );
  m_traitsvec[FRAME_CENTER_LONGITUDE] =      NEW_TRAIT( double,       "Frame center longitude" );
  m_traitsvec[UPPER_LEFT_CORNER_LAT] =       NEW_TRAIT( double,       "Upper left corner latitude" );
  m_traitsvec[UPPER_LEFT_CORNER_LON] =       NEW_TRAIT( double,       "Upper left corner longitude" );
  m_traitsvec[UPPER_RIGHT_CORNER_LAT] =      NEW_TRAIT( double,       "Upper right corner latitude" );
  m_traitsvec[UPPER_RIGHT_CORNER_LON] =      NEW_TRAIT( double,       "Upper right corner longitude" );
  m_traitsvec[LOWER_RIGHT_CORNER_LAT] =      NEW_TRAIT( double,       "Lower right corner latitude" );
  m_traitsvec[LOWER_RIGHT_CORNER_LON] =      NEW_TRAIT( double,       "Lower right corner longitude" );
  m_traitsvec[LOWER_LEFT_CORNER_LAT] =       NEW_TRAIT( double,       "Lower left corner latitude" );
  m_traitsvec[LOWER_LEFT_CORNER_LON] =       NEW_TRAIT( double,       "Lower left corner longitude" );
  m_traitsvec[SLANT_RANGE] =                 NEW_TRAIT( double,       "Slant range" );
  m_traitsvec[ANGLE_TO_NORTH] =              NEW_TRAIT( double,       "Angle to north" );
  m_traitsvec[OBLIQUITY_ANGLE] =             NEW_TRAIT( double,       "Obliquity angle" );
  m_traitsvec[START_DATE_TIME_UTC] =         NEW_TRAIT( std::string,  "Start Date Time - UTC" );
  m_traitsvec[EVENT_START_DATE_TIME_UTC] =   NEW_TRAIT( std::string,  "Event Start Date Time - UTC" );
  m_traitsvec[MISSION_START_TIME] =          NEW_TRAIT( std::string,  "Mission Start Date Time - UTC" );
  m_traitsvec[UNIX_TIMESTAMP] =              NEW_TRAIT( uint64_t,     "Unix timestamp" );
  m_traitsvec[PLATFORM_TRUE_AIRSPEED] =      NEW_TRAIT( double,       "Platform true airspeed" );
  m_traitsvec[PLATFORM_INDICATED_AIRSPEED] = NEW_TRAIT( double,       "Platform indicated airspeed" );
  m_traitsvec[PLATFORM_CALL_SIGN] =          NEW_TRAIT( std::string,  "Platform call sign" );
  m_traitsvec[FOV_NAME] =                    NEW_TRAIT( std::string,  "Field of view name" );
  m_traitsvec[WIND_DIRECTION] =              NEW_TRAIT( std::string,  "Wind Direction" );
  m_traitsvec[WIND_SPEED] =                  NEW_TRAIT( double,       "Wind Speed" );
  m_traitsvec[PREDATOR_UAV_UMS] =            NEW_TRAIT( std::string,  "Predator UAV Universal Metadata Set" );
  m_traitsvec[PREDATOR_UAV_UMS_V2] =         NEW_TRAIT( std::string,  "Predator UAV Universal Metadata Set v2.0" );
  m_traitsvec[SENSOR_RELATIVE_ROLL_ANGLE] =  NEW_TRAIT( double,       "Sensor Relative Roll Angle" );
  m_traitsvec[MISSION_ID] =                  NEW_TRAIT( std::string,  "Mission ID" );
  m_traitsvec[PLATFORM_TAIL_NUMBER] =        NEW_TRAIT( std::string,  "Platform tail number" );
  m_traitsvec[MISSION_NUMBER] =              NEW_TRAIT( std::string,  "Episode Number" );
  m_traitsvec[ANGLE_TO_NORTH] =              NEW_TRAIT( double,       "Angle to North" );
  m_traitsvec[OBLIQUITY_ANGLE] =             NEW_TRAIT( double,       "Sensor Elevation Angle" );
  m_traitsvec[SENSOR_ROLL_ANGLE] =           NEW_TRAIT( double,       "Sensor Roll Angle" );

#undef NEW_TRAIT

}


// ------------------------------------------------------------------
klv_0104::~klv_0104()
{
  for ( unsigned int i = 0; i < m_traitsvec.size(); i++ )
  {
    delete m_traitsvec[i];
    m_traitsvec[i] = 0;
  }
}


// ------------------------------------------------------------------
template < class T >
std::string
traits< T >::to_string( kwiver::vital::any const& data ) const
{
  T var = kwiver::vital::any_cast< T > ( data );
  std::stringstream ss;

  ss << var;
  return ss.str();
}


// ------------------------------------------------------------------
template < class T >
kwiver::vital::any
traits< T >::convert( uint8_t const* data, std::size_t length )
{
  if ( sizeof( T ) != length )
  {
    static kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.convert_0104_metadata" ) );
    char const* cp = reinterpret_cast< char const* >(data);
    std::string msg(cp, length);
    std::stringstream os;

    for ( unsigned int k = 0; k < msg.size(); ++k )
    {
      os  << std::hex << std::setfill( '0' ) << std::setw( 2 )
          << static_cast< unsigned int > ( msg[k] );
    }

    LOG_DEBUG( logger, "Data length does not match type length.  Data length = "
              << length << ", sizeof(type) = " << sizeof( T )
              << " - " << os.str() );
  }

  union
  {
    T val;
    char bytes[sizeof( T )];
  } converter;

  for ( int i = sizeof( T ) - 1; i >= 0; i--, data++ )
  {
    converter.bytes[i] = *data;
  }

  return converter.val;
}


// ------------------------------------------------------------------
// Specialization for extracting strings from a raw byte stream
template < >
kwiver::vital::any
traits< std::string >::convert( uint8_t const* data, std::size_t length )
{
  std::string value( reinterpret_cast< char const* > ( data ), length );

  return value;
}


// ------------------------------------------------------------------
//Handle real values as floats or doubles but return double
template < >
kwiver::vital::any
traits< double >::convert( uint8_t const* data, std::size_t length )
{
  double val;

  if ( length == sizeof( float ) )
  {
    union
    {
      float val;
      char bytes[sizeof( float )];
    } converter;

    for ( int i = sizeof( float ) - 1; i >= 0; i--, data++ )
    {
      converter.bytes[i] = *data;
    }

    val = static_cast< double > ( converter.val );
  }
  else if ( length == sizeof( double ) )
  {
    union
    {
      double val;
      char bytes[sizeof( double )];
    } converter;

    for ( int i = sizeof( double ) - 1; i >= 0; i--, data++ )
    {
      converter.bytes[i] = *data;
    }

    val = converter.val;
  }
  else
  {
    throw kwiver::vital::klv_exception( "Length does not correspond to double or float." );
  }

  return val;
}


// ------------------------------------------------------------------
// Specialization for extracting std_0102_lds from a raw byte stream
template < >
kwiver::vital::any
traits< std_0102_lds >::convert( uint8_t const* data, std::size_t length )
{
  // TODO:  Need to handle this for real
  std::string value( reinterpret_cast< char const* > ( data ), length );

  return value;
}

// ------------------------------------------------------------------
  // TEMP handling for std_0102_lds objects.
template < >
std::string
traits< std_0102_lds >::to_string( kwiver::vital::any const& data ) const
{
  std::string var = kwiver::vital::any_cast< std::string > ( data );
  return var;
}


// ------------------------------------------------------------------
klv_0104::tag
klv_0104::get_tag( klv_uds_key const& k ) const
{
  std::map< klv_uds_key, tag >::const_iterator itr = m_key_to_tag.find( k );

  if ( itr == m_key_to_tag.end() )
  {
    return UNKNOWN;
  }

  return itr->second;
}


// ------------------------------------------------------------------
kwiver::vital::any
klv_0104::get_value( tag tg, uint8_t const* data, std::size_t length )
{
  return m_traitsvec[tg]->convert( data, length );
}


// ------------------------------------------------------------------
template < class T >
T
klv_0104::get_value( tag tg, kwiver::vital::any const& data ) const
{
  traits< T >* t = dynamic_cast< traits< T >* > ( m_traitsvec[tg] );
  if ( ! t )
  {
    throw kwiver::vital::klv_exception( t->m_name + "' type mismatch" );
  }

  return kwiver::vital::any_cast< T > ( data );
}


// ------------------------------------------------------------------
std::string
klv_0104::get_string( tag tg, kwiver::vital::any const& data ) const
{
  return m_traitsvec[tg]->to_string( data );
}


// ------------------------------------------------------------------
std::string
klv_0104::get_tag_name( tag tg ) const
{
  return m_traitsvec[tg]->m_name;
}


// ------------------------------------------------------------------
klv_0104::traits_base const&
klv_0104::get_traits( tag tg ) const
{
  return *m_traitsvec[tg];
}


// ------------------------------------------------------------------
klv_uds_key
klv_0104::key()
{
  return klv_0104_uds_key;
}


// ------------------------------------------------------------------
bool
klv_0104::is_key( klv_uds_key const& key )
{
  return key == klv_0104_uds_key;
}


template double klv_0104::get_value< double > ( tag tg, kwiver::vital::any const& data ) const;
template uint64_t klv_0104::get_value< uint64_t > ( tag tg, kwiver::vital::any const& data ) const;
template std::string klv_0104::get_value< std::string > ( tag tg, kwiver::vital::any const& data ) const;

} }   // end namespace
