// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "misp_time.h"

#include <cstddef>
#include <string>

namespace kwiver {
namespace vital {

namespace {

static const std::string misp_tag("MISPmicrosectime");

}

// ==================================================================
//Extract the time stamp from the buffer
  bool convert_MISP_microsec_time( std::vector< unsigned char > const& buf, std::int64_t& ts )
{
  enum MISP_time_code { MISPmicrosectime = 0,
                        FLAG = 16,
                        MSB_0,
                        MSB_1,
                        IGNORE_0,
                        MSB_2,
                        MSB_3,
                        IGNORE_1,
                        MSB_4,
                        MSB_5,
                        IGNORE_2,
                        MSB_6,
                        MSB_7,
                        IGNORE_3,
                        MISP_NUM_ELEMENTS };

  //Check that the tag is the first thing in buf
  for ( std::size_t i = 0; i < FLAG; i++ )
  {
    if ( buf[i] != misp_tag[i] )
    {
      return false;
    }
  }

  if ( buf.size() >= MISP_NUM_ELEMENTS )
  {
    ts = 0;

    ts |= static_cast< int64_t > ( buf[MSB_7] );
    ts |= static_cast< int64_t > ( buf[MSB_6] ) << 8;
    ts |= static_cast< int64_t > ( buf[MSB_5] ) << 16;
    ts |= static_cast< int64_t > ( buf[MSB_4] ) << 24;

    ts |= static_cast< int64_t > ( buf[MSB_3] ) << 32;
    ts |= static_cast< int64_t > ( buf[MSB_2] ) << 40;
    ts |= static_cast< int64_t > ( buf[MSB_1] ) << 48;
    ts |= static_cast< int64_t > ( buf[MSB_0] ) << 56;

    return true;
  }

  return false;
} // convert_MISP_microsec_time

// ------------------------------------------------------------------
  bool find_MISP_microsec_time(  std::vector< unsigned char > const& pkt_data, std::int64_t& ts )
{
  bool retval(false);

  //Check if the data packet has enough bytes for the MISPmicrosectime packet
  if ( pkt_data.size() < misp_tag.length() + 13 )
  {
    return false;
  }

  bool found;
  std::size_t ts_location = std::string::npos;
  std::size_t last = pkt_data.size() - misp_tag.size();
  for ( std::size_t i = 0; i <= last; i++ )
  {
    found = true;
    for ( std::size_t j = 0; j < misp_tag.size(); j++ )
    {
      if ( pkt_data[i + j] != misp_tag[j] )
      {
        found = false;
        break;
      }
    }

    if ( found )
    {
      ts_location = i;
      break;
    }
  } // end for

  if ( ( std::string::npos != ts_location )
       && ( ( ts_location + misp_tag.length() + 13 ) < pkt_data.size() ) )
  {
    std::vector< unsigned char > MISPtime_buf( pkt_data.begin() + ts_location,
                                               pkt_data.begin() + ts_location + misp_tag.length() + 13 );
    ts = 0;

    retval = convert_MISP_microsec_time( MISPtime_buf, ts );
  }

  return retval;
}

} } // end namespace
