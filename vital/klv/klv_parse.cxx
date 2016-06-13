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
 * \brief This file contains the implementation for the klv parser.
 */

#include "klv_parse.h"
#include "klv_data.h"
#include "klv_key.h"
#include "klv_0601.h"
#include "klv_0104.h"

#include <vital/exceptions/klv.h>

#include <vital/logger/logger.h>

#include <cctype>

namespace kwiver {
namespace vital {

namespace {

std::string
FormatString( std::string const& val )
{
  const char hex_chars[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                               '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
  const size_t len( val.size() );
  bool unprintable_found(false);
  std::string ascii;
  std::string hex;

  for (size_t i = 0; i < len; i++)
  {
    char const byte = val[i];
    if ( ! isprint( byte ) )
    {
      ascii.append( 1, '.' );
      unprintable_found = true;
    }
    else
    {
      ascii.append( 1, byte );
    }

    // format as hex
    if (i > 0)
    {
      hex += " ";
    }

    hex += hex_chars[ ( byte & 0xF0 ) >> 4 ];
    hex += hex_chars[ ( byte & 0x0F ) >> 0 ];

  } // end for

  if (unprintable_found)
  {
    ascii += " (" + hex + ")";
  }

  return ascii;
}

} // end namespace


// ----------------------------------------------------------------
/** Extract a KLV length using BER (basic encoding rules)
 *
 * @param[in] buffer the buffer of bytes to parse
 * @param[in] buffer_len the length of the buffer
 * @param[out] offset the number of bytes representing the length
 * @param[out] value_len the length: number of bytes representing the value
 *
 * @returns True if successful. False if invalid or insufficient data.
 */
template < class ITERATOR >
bool
klv_ber_length( ITERATOR buffer,
                unsigned int buffer_len,
                uint8_t& offset,
                unsigned int& value_len )
{
  // handle the short form with 1 byte length description, first bit is 0
  if ( ! ( 0x80 & *buffer ) )
  {
    offset = 1;
    value_len = *buffer;
    return true;
  }

  offset = ( 0x7F & *buffer ) + 1;

  if ( offset > 5 )
  {
    kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.klv_parse" ) );
    LOG_WARN( logger, "BER encoded length more then 4 bytes: "
               << static_cast< int > ( offset ) );
    return false;
  }

  if ( offset > buffer_len )
  {
    // not enough data in the buffer to fully parse length
    return false;
  }

  value_len = 0;
  for ( uint8_t i = 1; i < offset; ++i )
  {
    value_len <<= 8;
    value_len += *( buffer + i );
  }
  return true;
}


// ----------------------------------------------------------------
/** @brief Pop the first KLV UDS key-value pair found in the data buffer.
 *
 * The first valid KLV packet found in the data stream is returned.
 * Leading bytes that do not belong to a KLV pair are dropped. The
 * input byte stream is modified internally and unprocessed partial
 * packets are left there so more raw data can be added to the stream
 * and packet parsing can be attempted later.
 *
 * @param[in,out] data Byte stream to be parsed.
 * @param[out] klv_packet Full klv packet with key and data fields
 * specified.
 *
 * @return \c true if packet returned; \c false if no packet returned.
 */
bool
klv_pop_next_packet( std::deque< uint8_t >&  data,
                     klv_data&               klv_packet )
{
  const std::size_t klv_key_length = klv_uds_key::size();

  while ( data.size() > klv_key_length + 1 )
  {
    // The buffer must start with key prefix for best results.
    // Would be nice if this could be done by method call
    // if (uds_key::matches_prefix(data) ) ...
    if ( ( data[0] == klv_uds_key::prefix[0] ) &&
         ( data[1] == klv_uds_key::prefix[1] ) &&
         ( data[2] == klv_uds_key::prefix[2] ) &&
         ( data[3] == klv_uds_key::prefix[3] ) )
    {
      // This is a little wierd, but it is this approach or I write
      // another constructor signature. The data needs to be copied
      // from the dqueue one element at a time because &data[1] is not
      // always equal to (&data[0] +1) due to memory management
      // practices in the dqueue (although sometimes it is).
      //
      // We are guaranteed enough bytes in the dqueue because of the
      // preceeding test in while()
      uint8_t temp[16];
      for ( int i = 0; i < 16; ++i )
      {
        temp[i] = data[i];
      } // end for

      klv_uds_key temp_key( temp );

      if ( temp_key.is_valid() )
      {
        if ( temp_key.category() == klv_uds_key::CATEGORY_LABEL )
        {
          // collect bytes that make up the key
          klv_data::container_t raw_data( data.begin(), data.begin() + klv_key_length );
          klv_packet = klv_data( raw_data, 0, klv_key_length, 0, 0 );

          // Keys with category "Label" have no length or value data
          data.erase( data.begin(), data.begin() + klv_key_length );
          return true;
        }

        uint8_t offset;
        unsigned int length;
        if ( klv_ber_length( data.begin() + klv_key_length,
                             data.size() - klv_key_length,
                             offset, length ) )
        {
          size_t total_len = klv_key_length + offset + length;

          // Is the full packet in the input buffer?
          if ( data.size() >= total_len )
          {
            klv_data::container_t raw_data( data.begin(), data.begin() + total_len ); // extract the key bytes
            klv_packet = klv_data( raw_data,       // total raw data
                                   0,              // key offset
                                   klv_key_length, // length of key in bytes
                                   klv_key_length + offset, // value offset (start of value bytes)
                                   length );                // length of value in bytes

            data.erase( data.begin(), data.begin() + total_len );
            return true;
          }
        }
        break;
      }

    } // end valid key

    // If prefix does not match or key not valid
    // Delete byte from top of input and try again
    kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.klv_parse" ) );
    LOG_DEBUG( logger, "discarding klv byte - 0x" << std::hex << int(data[0])
               //  << " rest of prefix: 0x"
               // << int(data[1]) << " 0x"
               // << int(data[2]) << " 0x"
               // << int(data[3]) << " -- 0x"
               // << int(data[4]) << " 0x"
               // << int(data[5]) << " 0x"
               // << int(data[6]) << " 0x"
               // << int(data[7])
             );
    data.pop_front();

  } // end while

  return false;
} // pop_klv_uds_pair


// ----------------------------------------------------------------
/** Parse out Local Data Set (LDS) packet.
 *
 * The data portion of the raw KLV packet is parsed into LDS packets.
 */
std::vector< klv_lds_pair >
parse_klv_lds( klv_data const& data )
{
  std::vector< klv_lds_pair > lds_pairs;
  uint8_t offset;
  unsigned int value_len;
  size_t len = data.value_size();
  klv_data::const_iterator_t it = data.value_begin();

  while ( ( len > 3 ) &&
          klv_ber_length( it + 1, len - 1, offset, value_len ) &&
          ( offset + 1 + value_len <= len ) )
  {
    klv_lds_key key( *it ); // one byte key
    std::vector< uint8_t > value( it + offset + 1, it + offset + 1 + value_len );
    lds_pairs.push_back( klv_lds_pair( key, value ) );

    // update pointer into data
    it = it + 1 + offset + value_len;
    len -= 1 + offset + value_len;
  }

  if ( len != 0 )
  {
    kwiver::vital::logger_handle_t logger( kwiver::vital::get_logger( "vital.klv_parse" ) );
    LOG_WARN( logger, len << " bytes left over when parsing LDS" );
  }

  return lds_pairs;
}


// ----------------------------------------------------------------
/** Parse data set with universal keys */
klv_uds_vector_t
parse_klv_uds( klv_data const& data )
{
  std::vector< klv_uds_pair > uds_pairs;

  klv_data pk;

  // create queue of data portion of packet.
  std::deque< uint8_t > deq( data.value_begin(), data.value_end() );

  while ( klv_pop_next_packet( deq, pk ) )
  {
    klv_uds_key uds_key( pk ); // 16 byte key
    uds_pairs.push_back( klv_uds_pair( uds_key,
         std::vector< uint8_t > ( pk.value_begin(), pk.value_end() ) ) );
  }

  return uds_pairs;
}


// ----------------------------------------------------------------
std::ostream&
print_klv( std::ostream& str, klv_data const& klv )
{
  klv_uds_key uds_key( klv ); // create key from raw data

  if ( is_klv_0601_key( uds_key ) )
  {
    str << "0601 Universal Key of size " << klv.value_size() << std::endl;
    if ( ! klv_0601_checksum( klv ) )
    {
      str << "checksum failed" << std::endl;
      str << "Raw hex of packet: " << klv << std::endl;
    }

    // Try to decode even if checksum failed.
    // This is useful when a valid packet has a bad checksum.
    // May fail badly if packet is really corrupt.
    klv_lds_vector_t lds = parse_klv_lds( klv );

    str << "  found " << lds.size() << " tags" << std::endl;
    for ( auto itr = lds.begin(); itr != lds.end(); ++itr )
    {
      if ( ( itr->first <= KLV_0601_UNKNOWN ) || ( itr->first >= KLV_0601_ENUM_END ) )
      {
        str << "    #" << int(itr->first) << " is not supported" << std::endl;
        continue;
      }

      // Convert a single tag
      const klv_0601_tag tag( klv_0601_get_tag( itr->first ) ); // get tag code from key

      // Extract relevant data from associated data bytes.
      kwiver::vital::any data = klv_0601_value( tag,
                                                &itr->second[0], itr->second.size() );

      str << "    #" << tag << " - "
          << klv_0601_tag_to_string( tag )
          << ": " << klv_0601_value_string( tag, data ) << " "
          << " [" << klv_0601_value_hex_string( tag, data ) << "]"
          << std::endl;
    } // end for
  }
  else if ( klv_0104::is_key( uds_key ) )
  {
    str << "Predator (0104) Universal Key of size " << klv.value_size() << std::endl;

    klv_uds_vector_t uds = parse_klv_uds( klv );
    str << "  found " << uds.size() << " tags" << std::endl;

    // Vector has key and data
    for ( auto itr = uds.begin(); itr != uds.end(); ++itr )
    {
      try
      {
        klv_0104::tag tag = klv_0104::instance()->get_tag( itr->first );
        if ( tag == klv_0104::UNKNOWN )
        {
          str << "Unknown key: " << itr->first << "Length: " << itr->second.size() << " bytes\n";
          continue;
        }

        kwiver::vital::any data = klv_0104::instance()->get_value( tag, &itr->second[0], itr->second.size() );
        std::string str_val = FormatString( klv_0104::instance()->get_string( tag, data ) );

        str << "    #" << tag << " - "
            << klv_0104::instance()->get_tag_name( tag )
            << "(" <<  itr->second.size() << " bytes): " << str_val << " "
            << std::endl;

      }
      catch ( kwiver::vital::klv_exception const& e )
      {
        str << "Error in 0104 klv: " << e.what() << "\n";
      }
    } // end for
  }
  else
  {
    str << "Unsupported UDS Key: " << uds_key << " data size is "
        << klv.value_size() << std::endl;

    switch ( uds_key.category() )
    {
    case klv_uds_key::CATEGORY_SINGLE:
      str << "  Contains a single data item." << std::endl;
      break;

    case klv_uds_key::CATEGORY_GROUP:
    {
      switch ( uds_key.group_type() )
      {
      case klv_uds_key::GROUP_UNIVERSAL_SET:
        str << "  Contains a universal set." << std::endl;
        break;

      case klv_uds_key::GROUP_GLOBAL_SET:
        str << "  Contains a global set." << std::endl;
        break;

      case klv_uds_key::GROUP_LOCAL_SET:
      {
        str << "  Contains a local set." << std::endl;
        typedef std::vector< klv_lds_pair > lds_data_t;
        lds_data_t lds = parse_klv_lds( klv );
        str << "    found " << lds.size() << " tags" << std::endl;
        str << "    local keys:";
        for ( lds_data_t::const_iterator itr = lds.begin(); itr != lds.end(); ++itr )
        {
          str << " " << itr->first;
        }
        str << std::endl;
        break;
      }

      case klv_uds_key::GROUP_VARIABLE_PACK:
        str << "  Contains a variable length pack." << std::endl;
        break;

      case klv_uds_key::GROUP_FIXED_PACK:
        str << "  Contains a fixed length pack." << std::endl;
        break;

      default:
        str << "  Contains an invalid type of group." << std::endl;
        break;
      }
      break;
    }

    case klv_uds_key::CATEGORY_WRAPPER:
      str << "  Is a wrapper around another data format." << std::endl;
      break;

    case klv_uds_key::CATEGORY_LABEL:
      str << "  Is a label and contains no data." << std::endl;
      break;

    default:
      str << "  Format is unknown." << std::endl;
      break;
    }        // switch
  }

  return str;
} // print_klv


} }   // end namespace
