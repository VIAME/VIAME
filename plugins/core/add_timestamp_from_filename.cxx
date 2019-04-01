/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "add_timestamp_from_filename.h"

#include <vital/algo/algorithm_factory.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <kwiversys/SystemTools.hxx>

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
add_timestamp_from_filename::add_timestamp_from_filename()
{
  this->set_capability( kwiver::vital::algo::image_io::HAS_TIME, true );
}

// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
  add_timestamp_from_filename::get_configuration() const
{
  auto config = kwiver::vital::algo::image_io::get_configuration();

  kwiver::vital::algo::image_io::get_nested_algo_configuration(
    "image_reader", config, this->image_reader);

  return config;
}

// ----------------------------------------------------------------------------
void add_timestamp_from_filename::set_configuration(
  kwiver::vital::config_block_sptr config )
{
  auto new_config = this->get_configuration();
  new_config->merge_config( config );

  kwiver::vital::algo::image_io::set_nested_algo_configuration(
    "image_reader", new_config, this->image_reader );
}

// ----------------------------------------------------------------------------
bool add_timestamp_from_filename::check_configuration(
  kwiver::vital::config_block_sptr config ) const
{
  return kwiver::vital::algo::image_io::check_nested_algo_configuration(
    "image_reader", config );
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr add_timestamp_from_filename::load_(
  std::string const& filename ) const
{
  if( this->image_reader )
  {
    auto im = this->image_reader->load( filename );
    im->set_metadata( this->fixup_metadata( filename, im->get_metadata() ) );
    return im;
  }

  return nullptr;
}

// ----------------------------------------------------------------------------
void add_timestamp_from_filename::save_(
  std::string const& filename,
  kwiver::vital::image_container_sptr data ) const
{
  if( this->image_reader )
  {
    this->image_reader->save( filename, data );
  }
}

// ----------------------------------------------------------------------------
kwiver::vital::metadata_sptr add_timestamp_from_filename::load_metadata_(
  std::string const& filename) const
{
  if( this->image_reader )
  {
    return this->fixup_metadata( filename,
      this->image_reader->load_metadata( filename ) );
  }

  return this->fixup_metadata( filename, nullptr );
}

// ----------------------------------------------------------------------------
std::vector< std::string > split( const std::string &s, char delim )
{
  std::stringstream ss( s );
  std::string item;
  std::vector<std::string> elems;

  while( std::getline( ss, item, delim ) )
  {
    elems.push_back(item);
  }
  return elems;
}

// ----------------------------------------------------------------------------
kwiver::vital::metadata_sptr add_timestamp_from_filename::fixup_metadata(
  std::string const& filename, kwiver::vital::metadata_sptr md ) const
{
  if( !md )
  {
    md = std::make_shared<kwiver::vital::metadata>();
  }

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

      kwiver::vital::time_usec_t msec =
        std::stoi( parts[4].substr( 7, 3 ) ) * 1e3;
      utc_time_usec =
        static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + msec;
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

        kwiver::vital::time_usec_t msec =
          std::stoi( date_str.substr( 15, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + msec;
      }
    }
    else if( parts.size() == 1 )
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

        kwiver::vital::time_usec_t msec =
          std::stoi( parts[2].substr( 0, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + msec;
      }
      // Example 201503.20150517.105551974.76450.png
      if( parts.size() > 3 && parts[0].size() == 6 && parts[1].size() == 8 )
      {
        tm t;

        t.tm_year = std::stoi( parts[1].substr( 0, 4 ) ) - 1900;
        t.tm_mon = std::stoi( parts[1].substr( 4, 2 ) ) - 1;
        t.tm_mday = std::stoi( parts[1].substr( 6, 2 ) );
  
        t.tm_hour = std::stoi( parts[2].substr( 0, 2 ) );
        t.tm_min = std::stoi( parts[2].substr( 2, 2 ) );
        t.tm_sec = std::stoi( parts[2].substr( 4, 2 ) );

        kwiver::vital::time_usec_t msec =
          std::stoi( parts[2].substr( 6, 3 ) ) * 1e3;
        utc_time_usec =
          static_cast< kwiver::vital::time_usec_t >( timegm( &t ) ) * 1e6 + msec;
      }
    }
  }

  if( !utc_time_usec )
  {
    throw std::runtime_error( "Unable to decode timestamp for file: " + filename );
  }

  kwiver::vital::timestamp ts;
  ts.set_time_usec( utc_time_usec );
  md->set_timestamp( ts );

  return md;
}

}
