/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
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

#include "filename_to_timestamp.h"

#include <kwiversys/SystemTools.hxx>

#include <cctype>
#include <locale>
#include <exception>
#include <sstream>
#include <string>
#include <ctime>
#include <regex>

#ifdef WIN32
#define timegm _mkgmtime
#endif

namespace viame
{

// ----------------------------------------------------------------------------
std::vector< std::string > split( const std::string &s, char delim )
{
  std::stringstream ss( s );
  std::string item;
  std::vector< std::string > elems;

  while( std::getline( ss, item, delim ) )
  {
    elems.push_back(item);
  }
  return elems;
}

// ----------------------------------------------------------------------------
std::vector< std::string > split( const std::string &s, std::string delims )
{
  std::vector< std::string > output( 1, s );

  for( unsigned c = 0; c < s.size(); c++ )
  {
    char delim = delims[c];

    std::vector< std::string > new_output;

    for( auto old_string : output )
    {
      for( auto to_add : split( old_string, delim ) )
      {
        new_output.push_back( to_add );
      }
    }

    output = new_output;
  }

  return output;
}

// ----------------------------------------------------------------------------
kwiver::vital::time_usec_t
convert_to_timestamp( const std::string& filename, const bool auto_discover )
{
  kwiver::vital::time_usec_t utc_time_usec = 0;

  if( filename.size() > 10 )
  {
    std::string name_only = kwiversys::SystemTools::GetFilenameName( filename );
    std::vector< std::string > parts = split( name_only, '_' );

    // Example: CHESS_FL1_C_160407_234502.428_COLOR-8-BIT.JPG
    if( parts.size() > 4 && parts[0] == "CHESS" &&
        parts[3].size() > 5 && parts[4].size() > 9 )
    {
      tm t;

      t.tm_year = 100 + std::stoi( parts[3].substr( 0, 2 ) );
      t.tm_mon = std::stoi( parts[3].substr( 2, 2 ) ) - 1;
      t.tm_mday = std::stoi( parts[3].substr( 4, 2 ) );

      t.tm_hour = std::stoi( parts[4].substr( 0, 2 ) );
      t.tm_min = std::stoi( parts[4].substr( 2, 2 ) );
      t.tm_sec = std::stoi( parts[4].substr( 4, 2 ) );

      kwiver::vital::time_usec_t usec =
        std::stoi( parts[4].substr( 7, 3 ) ) * 1e3;
      utc_time_usec =
        static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
    }
    // Example: CHESS2016_N94S_FL23_P__20160518012412.111GMT_THERM-16BIT.PNG
    else if( parts.size() > 5 && parts[0].size() > 5 &&
             parts[0].substr( 0, 5 ) == "CHESS" )
    {
      std::string date_str = ( parts[4].empty() ? parts[5] : parts[4] );

      if( date_str.size() >= 21 && date_str.substr( 18, 3 ) == "GMT" )
      {
        tm t;

        t.tm_year = std::stoi( date_str.substr( 0, 4 ) ) - 1900;
        t.tm_mon = std::stoi( date_str.substr( 4, 2 ) ) - 1;
        t.tm_mday = std::stoi( date_str.substr( 6, 2 ) );

        t.tm_hour = std::stoi( date_str.substr( 8, 2 ) );
        t.tm_min = std::stoi( date_str.substr( 10, 2 ) );
        t.tm_sec = std::stoi( date_str.substr( 12, 2 ) );

        kwiver::vital::time_usec_t usec =
          std::stoi( date_str.substr( 15, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
      }
    }
    // Example: *_20190507_004346.455104* or *_20190401_220727.714*
    else if( parts.size() > 2 )
    {
      for( unsigned i = 0; i < parts.size()-1; i++ )
      {
        if( parts[i].size() == 8 &&
            parts[i][0] == '2' && // Invalid in year 3000, lol, I'll be dead. Maybe O_o.
            parts[i+1].size() >= 10 && parts[i+1][6] == '.' )
        {
          tm t;

          t.tm_year = std::stoi( parts[i].substr( 0, 4 ) ) - 1900;
          t.tm_mon = std::stoi( parts[i].substr( 4, 2 ) ) - 1;
          t.tm_mday = std::stoi( parts[i].substr( 6, 2 ) );

          t.tm_hour = std::stoi( parts[i+1].substr( 0, 2 ) );
          t.tm_min = std::stoi( parts[i+1].substr( 2, 2 ) );
          t.tm_sec = std::stoi( parts[i+1].substr( 4, 2 ) );

          kwiver::vital::time_usec_t usec;

          if( parts[i+1].size() < 12 )
          {
            usec = std::stoi( parts[i+1].substr( 7, 3 ) ) * 1e3;
          }
          else
          {
            usec = std::stoi( parts[i+1].substr( 7, 6 ) );
          }

          utc_time_usec = static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
          break;
        }
      }
    }

    if( !utc_time_usec )
    {
      parts = split( name_only, '.' );

      // Example: 20151023.200145.662.017459.png
      if( parts.size() > 3 && parts[0].size() == 8 && parts[1].size() == 6 )
      {
        tm t;

        t.tm_year = std::stoi( parts[0].substr( 0, 4 ) ) - 1900;
        t.tm_mon = std::stoi( parts[0].substr( 4, 2 ) ) - 1;
        t.tm_mday = std::stoi( parts[0].substr( 6, 2 ) );
  
        t.tm_hour = std::stoi( parts[1].substr( 0, 2 ) );
        t.tm_min = std::stoi( parts[1].substr( 2, 2 ) );
        t.tm_sec = std::stoi( parts[1].substr( 4, 2 ) );

        kwiver::vital::time_usec_t usec =
          std::stoi( parts[2].substr( 0, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
      }
      // Example: 00231.00232.20171025.182621.170.004021.tif
     else if( parts.size() > 6 && parts[2].size() == 8 && parts[3].size() == 6 )
      {
        tm t;

        t.tm_year = std::stoi( parts[2].substr( 0, 4 ) ) - 1900;
        t.tm_mon = std::stoi( parts[2].substr( 4, 2 ) ) - 1;
        t.tm_mday = std::stoi( parts[2].substr( 6, 2 ) );

        t.tm_hour = std::stoi( parts[3].substr( 0, 2 ) );
        t.tm_min = std::stoi( parts[3].substr( 2, 2 ) );
        t.tm_sec = std::stoi( parts[3].substr( 4, 2 ) );

        kwiver::vital::time_usec_t usec =
          std::stoi( parts[4].substr( 0, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
      }
      // Example 201503.20150517.105551974.76450.png
      else if( parts.size() > 3 && parts[0].size() == 6 && parts[1].size() == 8 )
      {
        tm t;

        t.tm_year = std::stoi( parts[1].substr( 0, 4 ) ) - 1900;
        t.tm_mon = std::stoi( parts[1].substr( 4, 2 ) ) - 1;
        t.tm_mday = std::stoi( parts[1].substr( 6, 2 ) );
  
        t.tm_hour = std::stoi( parts[2].substr( 0, 2 ) );
        t.tm_min = std::stoi( parts[2].substr( 2, 2 ) );
        t.tm_sec = std::stoi( parts[2].substr( 4, 2 ) );

        kwiver::vital::time_usec_t usec =
          std::stoi( parts[2].substr( 6, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
      }
      else if( auto_discover ) // Match known formats first then rely on auto
      {
        parts = split( name_only, "._+-" );

        // Okay we have not seen this timestamp type before. Can we autodetect
        // a new timestamp variant given a couple heuristics.
        std::vector< std::string > parsed_digits( 1 );
        std::string full_joint_number, after_group_number, unused_number;

        bool new_break = false;
        bool any_numbers = false;

        for( unsigned i = 0; i < filename.size(); ++i )
        {
          if( std::isdigit( filename[i] ) )
          {
            parsed_digits.back() += filename[i];
            full_joint_number += filename[i];

            new_break = true;
            any_numbers = true;
          }
          else if( new_break )
          {
            parsed_digits.push_back( std::string() );
            new_break = false;
          }
        }

        if( any_numbers )
        {
          int eight_position = -1;
          int six_position = -1;

          for( int i = 0; i < static_cast< int >( parsed_digits.size() ); ++i )
          {
            if( parsed_digits[i].size() == 8 && eight_position < 0 )
            {
              eight_position = i;
            }
            else if( parsed_digits[i].size() == 6 && six_position < 0 )
            {
              six_position = i;
            }
            else
            {
              if( six_position >= 0 && eight_position >= 0 )
              {
                after_group_number += parsed_digits[i];
              }
              unused_number += parsed_digits[i];
            }
          }

          // Top choice, maybe we have a date code or seconds string
          if( eight_position >= 0 )
          {
            const std::string& digit8 = parsed_digits[ eight_position ];

            tm t;

            t.tm_year = std::stoi( digit8.substr( 0, 4 ) ) - ( digit8[0] == '2' ? 1900 : 0 );
            t.tm_mon = std::stoi( digit8.substr( 4, 2 ) ) - 1;
            t.tm_mday = std::stoi( digit8.substr( 6, 2 ) );

            kwiver::vital::time_usec_t usec = 0;

            if( six_position >= 0 )
            {
              // This might actually be a legit timestamp and not a crapshoot
              const std::string& digit6 = parsed_digits[ six_position ];

              t.tm_hour = std::stoi( digit6.substr( 0, 2 ) );
              t.tm_min = std::stoi( digit6.substr( 2, 2 ) );
              t.tm_sec = std::stoi( digit6.substr( 4, 2 ) );

              if( after_group_number.size() >= 3 )
              {
                usec = std::stoi( after_group_number.substr( 0, 3 ) ) * 1e3;
              }
              else if( !unused_number.empty() )
              {
                usec = std::stoi( unused_number );
              }
            }
            else if( after_group_number.size() >= 9 )
            {
              t.tm_hour = std::stoi( after_group_number.substr( 0, 2 ) );
              t.tm_min = std::stoi( after_group_number.substr( 2, 2 ) );
              t.tm_sec = std::stoi( after_group_number.substr( 4, 2 ) );

              usec = std::stoi( after_group_number.substr( 6, 3 ) ) * 1e3;
            }
            else if( !unused_number.empty() )
            {
              usec = std::stoi( unused_number );
            }

            utc_time_usec =
              static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + usec;
          }
          else
          {
            // Last choice, use all numbers, likely incorrect but likely unique
            if( full_joint_number.size() > 9 )
            {
              full_joint_number = full_joint_number.substr( full_joint_number.size() - 9 );
            }

            utc_time_usec = std::stoi( full_joint_number );
          }
        }
      }
    }
  }

  if( !utc_time_usec )
  {
    throw std::runtime_error( "Unable to decode timestamp for file: " + filename );
  }

  return utc_time_usec;
}

}
