/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
