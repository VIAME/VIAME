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

#include "klv_0601.h"
#include "klv_0601_traits.h"
#include "klv_data.h"

#include <vital/logger/logger.h>

#include <typeinfo>
#include <cstring>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <functional>

namespace kwiver {
namespace vital {

namespace {

// A function type that converts raw byte streams to kwiver::vital::any
typedef std::function< kwiver::vital::any( const uint8_t*, std::size_t ) > klv_decode_func_t;

// ------------------------------------------------------------------
// Parse type T from a raw byte stream in MSB (most significant byte first) order
template < typename T >
kwiver::vital::any
klv_convert( const uint8_t* data, std::size_t length )
{
  if ( sizeof( T ) != length )
  {
    kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.klv_0601" ) );
    LOG_DEBUG( logger, "Data type (" <<  sizeof(T) << " bytes) and length ("
              << length << " bytes) differ in size." );
  }

  T value = *( data++ );
  for ( std::size_t i = 1; i < length; ++i, ++data )
  {
    value <<= 8;
    value |= *data;
  }

  return value;
}


// ------------------------------------------------------------------
// Specialization for extracting strings from a raw byte stream
template < >
kwiver::vital::any
klv_convert< std::string > ( const uint8_t* data, std::size_t length )
{
  std::string value( reinterpret_cast< const char* > ( data ), length );

  return value;
}


// ------------------------------------------------------------------
// Specialization for extracting STD 0102 LSD from raw byte stream
// \note this is a place holder for now.
template < >
kwiver::vital::any
klv_convert< kwiver::vital::std_0102_lds > ( const uint8_t* data, std::size_t length )
{
  // Need to decode this for real
  std::string value( reinterpret_cast< char const* > ( data ), length );

  return value;
}


// ------------------------------------------------------------------
// A function type that converts a kwiver::vital::any to a double
typedef std::function< double ( kwiver::vital::any const& ) > klv_any_to_double_func_t;


// ------------------------------------------------------------------
// Take a "convert T to double" function apply to a kwiver::vital::any
// This is used with boost bind to make a kwiver::vital::any to double conversion function
template < typename T >
double
klv_as_double( const std::function< double(T const& val) >& func,
               kwiver::vital::any const& data )
{
  return func( kwiver::vital::any_cast< T > ( data ) );
}


// A function type to format kwiver::vital::any raw data in hex and write to the ostream
typedef std::function< void ( std::ostream& os, kwiver::vital::any const& ) > klv_any_format_hex_func_t;


// ------------------------------------------------------------------
// Write kwiver::vital::any (with underlying type T) in hex
template < typename T >
void
format_hex( std::ostream& os, kwiver::vital::any const& data )
{
  std::iostream::fmtflags f( os.flags() );

  os  << std::hex << std::setfill( '0' ) << std::setw( sizeof( T ) * 2 )
      << kwiver::vital::any_cast< T > ( data );
  os.flags( f );
}


// ------------------------------------------------------------------
// Specialization for writing a byte in hex (so it doesn't print ASCII)
template < >
void
format_hex< uint8_t > ( std::ostream& os, kwiver::vital::any const& data )
{
  std::iostream::fmtflags f( os.flags() );

  os  << std::hex << std::setfill( '0' ) << std::setw( 2 )
      << static_cast< unsigned int > ( kwiver::vital::any_cast< uint8_t > ( data ) );
  os.flags( f );
}


// ------------------------------------------------------------------
// Specialization for writing a byte in hex (so it doesn't print ASCII)
template < >
void
format_hex< int8_t > ( std::ostream& os, kwiver::vital::any const& data )
{
  std::iostream::fmtflags f( os.flags() );

  os  << std::hex << std::setfill( '0' ) << std::setw( 2 )
      << static_cast< unsigned int > ( kwiver::vital::any_cast< int8_t > ( data ) );
  os.flags( f );
}


// ------------------------------------------------------------------
// Specialization for writing a string as a sequence of hex bytes
template < >
void
format_hex< std::string > ( std::ostream& os, kwiver::vital::any const& data )
{
  std::string s = kwiver::vital::any_cast< std::string > ( data );
  std::iostream::fmtflags f( os.flags() );

  for ( unsigned int k = 0; k < s.size(); ++k )
  {
    os  << std::hex << std::setfill( '0' ) << std::setw( 2 )
        << static_cast< unsigned int > ( s[k] );
  }
  os.flags( f );
}


// ------------------------------------------------------------------
// Specialization for writing a STD 0102 LDS in hex bytes
template < >
void
format_hex< kwiver::vital::std_0102_lds > ( std::ostream& os, kwiver::vital::any const& data )
{
  std::iostream::fmtflags f( os.flags() );
  std::string d = kwiver::vital::any_cast< std::string > ( data );

  for ( unsigned int k = 0; k < d.size(); ++k )
  {
    os  << std::hex << std::setfill( '0' ) << std::setw( 2 )
        << static_cast< unsigned int > ( d[k] );
  }
  os.flags( f );
}


// ------------------------------------------------------------------
// Store KLV 0601 traits for dynamic run-time lookup
// Build an array of these structs, one for each 0601 tag,
// using template metaprogramming.
struct klv_0601_dyn_traits
{
  std::string name;
  const std::type_info* type;
  unsigned int num_bytes;
  klv_decode_func_t decode_func;
  bool has_double;
  klv_any_to_double_func_t double_func;
  klv_any_format_hex_func_t any_hex_func;
};


// ------------------------------------------------------------------
// Recursive template metaprogram to populate the run-time array of traits
template < klv_0601_tag tag >
struct construct_traits
{
  // Populate the array element for this tag
  static inline std::vector< klv_0601_dyn_traits >&
  init( std::vector< klv_0601_dyn_traits >& data )
  {
    typedef typename klv_0601_traits< tag >::type type;
    klv_0601_dyn_traits& t = data[tag];
    t.name = klv_0601_traits< tag >::name();
    t.type =  &typeid( type );
    t.num_bytes = sizeof( type );
    t.decode_func = klv_decode_func_t( klv_convert< type > );
    t.has_double = klv_0601_convert< tag >::has_double;
    t.double_func = std::bind( klv_as_double< type >,
                                 klv_0601_convert< tag >::as_double, std::placeholders::_1 );
    t.any_hex_func = klv_any_format_hex_func_t( format_hex< type > );
    return construct_traits< klv_0601_tag( tag - 1 ) >::init( data );
  }


};


// ------------------------------------------------------------------
// The base case: unknown tag (with ID = 0)
template < >
struct construct_traits< KLV_0601_UNKNOWN >
{
  static inline std::vector< klv_0601_dyn_traits >& init( std::vector< klv_0601_dyn_traits >& data )
  {
    klv_0601_dyn_traits& t = data[KLV_0601_UNKNOWN];

    t.name = "Unknown";
    t.type =  &typeid( void );
    t.num_bytes = 0;
    t.has_double = false;
    return data;
  }
};


// ------------------------------------------------------------------
// Construct an array of traits for all known 0601 tags
std::vector< klv_0601_dyn_traits > init_traits_array()
{
  std::vector< klv_0601_dyn_traits > tmp( KLV_0601_ENUM_END );

  return construct_traits< klv_0601_tag( KLV_0601_ENUM_END - 1 ) >::init( tmp );
}


static const std::vector< klv_0601_dyn_traits > traits_array = init_traits_array();

static const uint8_t key_data[16] =
{
  0x06, 0x0e, 0x2b, 0x34,
  0x02, 0x0B, 0x01, 0x01,
  0x0E, 0x01, 0x03, 0x01,
  0x01, 0x00, 0x00, 0x00
};

static const klv_uds_key klv_0601_uds_key( key_data );

} // end anonymous namespace


//=============================================================================
// Public function implementations below
//=============================================================================

/** Globally available 0601 key.
 *
 */
klv_uds_key
klv_0601_key()
{
  return klv_0601_uds_key;
}


bool
is_klv_0601_key( klv_uds_key const& key )
{
  return key == klv_0601_uds_key;
}


klv_0601_tag
klv_0601_get_tag( klv_lds_key key )
{
  return static_cast< klv_0601_tag > ( uint8_t( key ) );
}

// ----------------------------------------------------------------
/** Compute 0601 checksum.
 *
 * Verify the 0601 block checksum against the expected checksum.  The
 * supplied data is a klv UDS packet which may contain multiple 0601
 * LDS data packets.
 *
 * The checksum is a running 16-bit sum through the entire UDS packet
 * starting with the 16 byte Local Data Set key and ending with
 * summing the length field of the checksum data item.
 *
 * Packet layout
 *
 *  |__| ... |__|__|__|__|
 *            ^  ^  ^--^----- 16 bit checksum
 *            |  |
 *            |   ----------- length of checksum in bytes (= 2)
 *             -------------- checksum packet type code (= 1)
 *
 * @param[in] data raw data to checksum
 *
 * @return \b True if checksum matches expected; \b False otherwise.
 * \b False is also returned if checksum tag is not found.
 */
bool
klv_0601_checksum( klv_data const& data )
{
  klv_data::const_iterator_t eit = data.klv_end();


  // if checksum tag is not where expected then terminate early
  if ( ( *( eit - 4 ) != 0x01 ) && //
       ( *( eit - 3 ) != 0x02 ) ) // cksum length
  {
    return false;
  }

  // Retrieve checksum from raw data
  uint16_t cksum = ( *( eit - 2 ) << 8 ) | ( *( eit - 1 ) );

  uint16_t bcc( 0 );
  size_t len = data.klv_size() - 2;
  klv_data::const_iterator_t cit = data.klv_begin();

  // Sum each 16-bit chunk within the buffer into a checksum
  for ( unsigned i = 0; i < len; i++ )
  {
    bcc += *cit << ( 8 * ( ( i + 1 ) % 2 ) );
    cit++;
  }

  return bcc == cksum;
}


// ------------------------------------------------------------------
// Return a string representation of the name of a KLV 0601 tag
std::string
klv_0601_tag_to_string( klv_0601_tag t )
{
  return traits_array[t].name;
}


// ------------------------------------------------------------------
// Extract the appropriate data type from raw bytes as a kwiver::vital::any
kwiver::vital::any
klv_0601_value( klv_0601_tag t, const uint8_t* data, std::size_t length )
{
  return traits_array[t].decode_func( data, length );
}


// ------------------------------------------------------------------
// Return the tag data as a double
double
klv_0601_value_double( klv_0601_tag t, kwiver::vital::any const& data )
{
  return traits_array[t].double_func( data );
}


// ------------------------------------------------------------------
// Return the tag data as a double
bool
klv_0601_has_double( klv_0601_tag t )
{
  return traits_array[t].has_double;
}


// ------------------------------------------------------------------
// Format the tag data as a string
std::string
klv_0601_value_string( klv_0601_tag t, kwiver::vital::any const& data )
{
  klv_0601_dyn_traits const& traits = traits_array[t];

  if ( traits.type == &typeid( std::string ) )
  {
    return kwiver::vital::any_cast< std::string > ( data );
  }

  if ( traits.has_double )
  {
    std::stringstream ss;
    ss  << std::setprecision( traits.num_bytes * 3 )
        << traits.double_func( data );
    return ss.str();
  }

  if ( t == KLV_0601_UNIX_TIMESTAMP )
  {
    std::stringstream ss;
    typedef klv_0601_traits< KLV_0601_UNIX_TIMESTAMP >::type time_type;

    ss << kwiver::vital::any_cast< time_type > ( data );
    return ss.str();
  }

  return "Unknown";
}


// ------------------------------------------------------------------
// Format the tag data as a hex string
std::string
klv_0601_value_hex_string( klv_0601_tag t, kwiver::vital::any const& data )
{
  std::stringstream ss;

  traits_array[t].any_hex_func( ss, data );
  return ss.str();
}

} } // end namespace
