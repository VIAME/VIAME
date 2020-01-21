/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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

// ----------------------------------------------------------------
static std::string source_name(size_t n)
{
  return "video_source_" + std::to_string(n);
}

// ----------------------------------------------------------------
class video_input_splice::priv
{
public:
  priv() :
  c_frame_skip( 1 ),
  d_has_timeout( false ),
  d_is_seekable( false ),
  d_frame_offset( 0 )
  { }

  // Configuration values
  unsigned int c_frame_skip;

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

  config->set_value( "output_nth_frame", d->c_frame_skip,
                     "Only outputs every nth frame of the video starting at the first frame. The output "
                     "of num_frames still reports the total frames in the video but skip_frame is valid "
                     "every nth frame only and there are metadata_map entries for only every nth frame.");

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
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d->c_frame_skip = config->get_value<vital::timestamp::frame_t>(
    "output_nth_frame", d->c_frame_skip );

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
    VITAL_THROW( kwiver::vital::invalid_file, list_name, "Could not open file" );
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
    if (!kwiversys::SystemTools::FileExists(filepath, true))
    {
      filepath = kwiversys::SystemTools::FindFile(filepath, d->d_search_path, true);
    }
    (*vs_iter)->open( filepath );
    ++vs_iter;
  }

  d->d_active_source = d->d_video_sources.begin();
  d->d_frame_offset = 0;

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
  for ( auto vs : d->d_video_sources )
  {
    if ( vs )
    {
      vs->close();
    }
  }

  d->d_metadata_map.clear();
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
  if ( d->d_active_source != d->d_video_sources.end() && *d->d_active_source )
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
  bool status = false;
  kwiver::vital::timestamp::frame_t frame_number = 1;

  if ( this->end_of_video() )
  {
    return status;
  }

  do
  {
    status = (*d->d_active_source)->next_frame(ts, timeout);
    frame_number = ts.get_frame() + d->d_frame_offset;

    if ( ! status )
    {
      // Move to next source if needed
      if ( (*d->d_active_source)->end_of_video() )
      {
        d->d_frame_offset += (*d->d_active_source)->num_frames();
        ++d->d_active_source;
        if ( d->d_active_source != d->d_video_sources.end() )
        {
          if ( (*d->d_active_source)->seekable() )
          {
            (*d->d_active_source)->seek_frame(ts, 1, timeout);
          }
          if ( ! (*d->d_active_source)->good() )
          {
            status = (*d->d_active_source)->next_frame(ts, timeout);
          }
          else
          {
            ts = (*d->d_active_source)->frame_timestamp();
            status = true;
          }
          frame_number = ts.get_frame() + d->d_frame_offset;
        }
      }
    }
  }
  while ( (frame_number - 1) % d->c_frame_skip != 0 && status );

  ts.set_frame( frame_number );
  return status;
} // video_input_splice::next_frame


// ------------------------------------------------------------------
bool
video_input_splice
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  using frame_t = kwiver::vital::timestamp::frame_t;
  bool status = false;

  // Check if requested frame would have been skipped
  if ( ( frame_number - 1 ) % d->c_frame_skip != 0 )
  {
    return false;
  }

  // Determine which source is responsible for this frame
  size_t frames_prior = 0;
  for ( auto vs_iter = d->d_video_sources.begin();
        vs_iter != d->d_video_sources.end();
        vs_iter++ )
  {
    if ( frame_number <= static_cast<frame_t>(
                           (*vs_iter)->num_frames() + frames_prior) )
    {
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
  if ( this->end_of_video() )
  {
    return {};
  }

  return (*d->d_active_source)->frame_timestamp();
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_splice
::frame_image()
{
  if ( this->end_of_video() )
  {
    return nullptr;
  }

  return (*d->d_active_source)->frame_image();
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_splice
::frame_metadata()
{
  if ( this->end_of_video() )
  {
    return {};
  }

  return (*d->d_active_source)->frame_metadata();
}


// ----------------------------------------------------------------
kwiver::vital::metadata_map_sptr
video_input_splice
::metadata_map()
{
  if ( d->d_metadata_map.empty() && d->d_video_sources.size() > 0 )
  {
    auto frame_offset = 0;
    for ( auto const& vs  : d->d_video_sources )
    {
      auto curr_metadata = vs->metadata_map()->metadata();
      for ( auto const& md : curr_metadata )
      {
        d->d_metadata_map.emplace(md.first + frame_offset, md.second);
      }
      frame_offset += vs->num_frames();
    }
  }
  return std::make_shared<kwiver::vital::simple_metadata_map>(d->d_metadata_map);
}

} } }     // end namespace
