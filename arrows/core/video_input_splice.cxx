/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "video_input_splice.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>

#include <kwiversys/SystemTools.hxx>

#include <fstream>

namespace kwiver {
namespace arrows {
namespace core {

std::string source_name(size_t n)
{
  return "video_source_" + std::to_string(n);
}

class video_input_splice::priv
{
public:
  priv() :
  d_has_timeout(false),
  d_is_seekable(false),
  d_frame_offset(0)
  { }

  std::vector< std::string > d_search_path;
  bool d_has_timeout;
  bool d_is_seekable;

  // Frame offset to get frame numbers correct
  kwiver::vital::timestamp::frame_t d_frame_offset;

  // Vector of video sources
  std::vector< vital::algo::video_input_sptr > d_video_sources;

  // Iterator to the active source
  std::vector< vital::algo::video_input_sptr >::iterator d_active_source;

  // Cached metadata map
  vital::metadata_map::map_metadata_t d_metadata_map;
};


// ------------------------------------------------------------------
video_input_splice
::video_input_splice()
  : d( new video_input_splice::priv )
{
  attach_logger( "video_input_splice" );
}


// ------------------------------------------------------------------
video_input_splice
::~video_input_splice()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
video_input_splice
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  size_t n = 1;
  for ( auto const& vs : d->d_video_sources )
  {
    vital::algo::video_input::
      get_nested_algo_configuration( source_name( n ), config, vs );
  }

  return config;
}


// ------------------------------------------------------------------
void
video_input_splice
::set_configuration( vital::config_block_sptr config )
{
  // Extract string and create vector of directories
  std::string path = config->get_value<std::string>( "path", "" );
  kwiver::vital::tokenize(
      path, d->d_search_path, ":", kwiver::vital::TokenizeTrimEmpty );
  d->d_search_path.push_back( "." ); // add current directory

  bool has_eov = true;
  bool has_frame_numbers = true;
  bool has_frame_data = true;
  bool has_frame_time = true;
  bool has_metadata = true;
  bool has_abs_fr_time = true;
  bool has_timeout = true;
  bool is_seekable = true;

  typedef vital::algo::video_input vi;

  size_t n = 1;
  vital::config_block_sptr source_config = config->subblock( source_name( n ) );

  while ( source_config->available_values().size() > 0 )
  {
    // Make sure the corresponding sources exists
    while ( d->d_video_sources.size() < n )
    {
      d->d_video_sources.push_back( vital::algo::video_input_sptr() );
    }

    vital::algo::video_input::set_nested_algo_configuration(
      source_name( n ), config, d->d_video_sources[n-1] );

    auto& caps = d->d_video_sources[n-1]->get_implementation_capabilities();

    has_eov = has_eov && caps.capability( vi::HAS_EOV );
    has_frame_numbers = has_frame_numbers && caps.capability( vi::HAS_FRAME_NUMBERS);
    has_frame_data = has_frame_data && caps.capability( vi::HAS_FRAME_DATA );
    has_frame_time = has_frame_time && caps.capability( vi::HAS_FRAME_TIME );
    has_metadata = has_metadata && caps.capability( vi::HAS_METADATA );
    has_abs_fr_time = has_abs_fr_time && caps.capability( vi::HAS_ABSOLUTE_FRAME_TIME);
    has_timeout = has_timeout && caps.capability( vi::HAS_TIMEOUT );
    is_seekable = is_seekable && caps.capability( vi::IS_SEEKABLE );

    ++n;
    source_config = config->subblock( source_name( n ) );
  }

  set_capability( vi::HAS_EOV, has_eov );
  set_capability( vi::HAS_FRAME_NUMBERS, has_frame_numbers );
  set_capability( vi::HAS_FRAME_DATA, has_frame_data );
  set_capability( vi::HAS_FRAME_TIME, has_frame_time );
  set_capability( vi::HAS_METADATA, has_metadata );
  set_capability( vi::HAS_ABSOLUTE_FRAME_TIME, has_abs_fr_time );
  set_capability( vi::HAS_TIMEOUT, has_timeout );
  set_capability( vi::IS_SEEKABLE, is_seekable );

  d->d_is_seekable = is_seekable;
  d->d_has_timeout = has_timeout;

  d->d_active_source = d->d_video_sources.begin();
}


// ------------------------------------------------------------------
bool
video_input_splice
::check_configuration( vital::config_block_sptr config ) const
{
  bool status = true;

  size_t n = 1;
  while ( config->has_value( source_name( n ) ) )
  {
    status = status && vital::algo::video_input::
      check_nested_algo_configuration( source_name( n ), config );
  }

  return status;
}


// ------------------------------------------------------------------
void
video_input_splice
::open( std::string list_name )
{
  // Close sources in case they are already open
  for ( auto& vs : d->d_video_sources)
  {
    vs->close();
  }

  // Open file and read lines
  std::ifstream ifs( list_name.c_str() );
  if ( ! ifs )
  {
    throw kwiver::vital::invalid_file( list_name, "Could not open file" );
  }

  // Add directory that contains the list file to the path
  std::string list_path = kwiversys::SystemTools::GetFilenamePath( list_name );
  if ( ! list_path.empty() )
  {
    d->d_search_path.push_back( list_path );
  }

  kwiver::vital::data_stream_reader stream_reader( ifs );
  auto vs_iter = d->d_video_sources.begin();
  std::string filepath;

  while ( stream_reader.getline( filepath ) && vs_iter != d->d_video_sources.end() )
  {
    if ( ! kwiversys::SystemTools::FileExists( filepath ) )
    {
      filepath = kwiversys::SystemTools::FindFile( filepath, d->d_search_path, true );
      if ( filepath.empty() )
      {
        throw kwiver::vital::
          file_not_found_exception( filepath, "could not locate file in path" );
      }
    }
    (*vs_iter)->open( filepath );
    ++vs_iter;
  }

  if ( vs_iter != d->d_video_sources.end() )
  {
    LOG_WARN( logger(), "Not enough entries in list file. Some of the video "
                        "source entries in the config file will not be used.");
  }

  if ( stream_reader.getline( filepath ) )
  {
    LOG_WARN( logger(), "Not enough video sources in config file. Some "
                        "entries from the list file will not be used.");
  }
}


// ------------------------------------------------------------------
void
video_input_splice
::close()
{
  // Close all the sources
  for (auto vs: d->d_video_sources)
  {
    if ( vs )
    {
      vs->close();
    }
  }

  d->d_metadata_map.clear();

  d->d_frame_offset = 0;
  d->d_has_timeout = false;
  d->d_is_seekable = false;
}


// ------------------------------------------------------------------
bool
video_input_splice
::end_of_video() const
{
  return ( d->d_active_source == d->d_video_sources.end() );
}


// ------------------------------------------------------------------
bool
video_input_splice
::good() const
{
  if ( *d->d_active_source )
  {
    return (*d->d_active_source)->good();
  }
  else
  {
    return false;
  }
}

// ------------------------------------------------------------------
bool
video_input_splice
::seekable() const
{
  return d->d_is_seekable;
}

// ------------------------------------------------------------------
size_t
video_input_splice
::num_frames() const
{
  size_t num_frames = 0;

  for ( auto& vs : d->d_video_sources )
  {
    num_frames += vs->num_frames();
  }

  return num_frames;
}

// ------------------------------------------------------------------
bool
video_input_splice
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout )
{
  // if timeout is not supported by both sources
  // then do not pass a time out value to either
  if ( ! d->d_has_timeout )
  {
    timeout = 0;
  }

  bool status;

  status = (*d->d_active_source)->next_frame(ts, timeout);

  if ( ! status )
  {
    // Move to next source if needed
    if ( (*d->d_active_source)->end_of_video() )
    {
      (*d->d_active_source)->seek_frame(ts, 1, timeout);
      d->d_frame_offset += (*d->d_active_source)->num_frames();
      d->d_active_source++;
      if ( d->d_active_source != d->d_video_sources.end() )
      {
        if ( ! (*d->d_active_source)->good() )
        {
          status = (*d->d_active_source)->next_frame(ts, timeout);
        }
        else
        {
          ts = (*d->d_active_source)->frame_timestamp();
          status = true;
        }
      }
    }
  }

  ts.set_frame( ts.get_frame() + d->d_frame_offset );
  return status;
} // video_input_splice::next_frame

// ------------------------------------------------------------------
bool
video_input_splice
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  bool status = false;
  // if timeout is not supported by all sources
  // then do not pass a time out value to active source
  if ( ! d->d_has_timeout )
  {
    timeout = 0;
  }

  // Determine which source is responsible for this frame
  size_t frames_prior = 0;
  for ( auto vs_iter = d->d_video_sources.begin();
        vs_iter != d->d_video_sources.end();
        vs_iter++ )
  {
    if ( frame_number <= (*vs_iter)->num_frames() + frames_prior )
    {
      (*d->d_active_source)->seek_frame( ts, 1, timeout );
      d->d_active_source = vs_iter;
      d->d_frame_offset = frames_prior;
      status =
        (*d->d_active_source)->seek_frame( ts, frame_number - frames_prior );
      break;
    }
    else
    {
      frames_prior += (*vs_iter)->num_frames();
    }
  }

  ts.set_frame( ts.get_frame() + d->d_frame_offset );
  return status;
} // video_input_splice::seek_frame

// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_splice
::frame_timestamp() const
{
  return (*d->d_active_source)->frame_timestamp();
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_splice
::frame_image()
{
  return (*d->d_active_source)->frame_image();
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_splice
::frame_metadata()
{
  return (*d->d_active_source)->frame_metadata();
}

kwiver::vital::metadata_map_sptr
video_input_splice
::metadata_map()
{
  if ( d->d_metadata_map.empty() && d->d_video_sources.size() > 0 )
  {
    auto frame_offset = 0;
    for ( auto vs_iter = d->d_video_sources.begin();
          vs_iter != d->d_video_sources.end();
          vs_iter++ )
    {
      auto curr_metadata = (*vs_iter)->metadata_map()->metadata();
      for ( auto const& md : curr_metadata )
      {
        d->d_metadata_map[md.first + frame_offset] = md.second;
      }
      frame_offset += (*vs_iter)->num_frames();
    }
  }
  return std::make_shared<kwiver::vital::simple_metadata_map>(d->d_metadata_map);
}

} } }     // end namespace
