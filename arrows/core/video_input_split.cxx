/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

#include "video_input_split.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>


namespace kwiver {
namespace arrows {
namespace core {

class video_input_split::priv
{
public:
  priv()
  : d_has_timeout( false )
  { }

  // local state
  bool d_has_timeout;

  // processing classes
  vital::algo::video_input_sptr d_image_source;
  vital::algo::video_input_sptr d_metadata_source;

};


// ------------------------------------------------------------------
video_input_split
::video_input_split()
  : d( new video_input_split::priv )
{
  attach_logger( "arrows.core.video_input_split" );
}


// ------------------------------------------------------------------
video_input_split
::~video_input_split()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
video_input_split
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  vital::algo::video_input::
    get_nested_algo_configuration( "image_source", config, d->d_image_source );

  vital::algo::video_input::
    get_nested_algo_configuration( "metadata_source", config, d->d_metadata_source );

  return config;
}


// ------------------------------------------------------------------
void
video_input_split
::set_configuration( vital::config_block_sptr config )
{
  vital::algo::video_input::
    set_nested_algo_configuration( "image_source", config, d->d_image_source );

  vital::algo::video_input::
    set_nested_algo_configuration( "metadata_source", config, d->d_metadata_source );
}


// ------------------------------------------------------------------
bool
video_input_split
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the image reader configuration.
  bool image_stat = vital::algo::video_input::
    check_nested_algo_configuration( "image_source", config );

  // Check the metadata reader configuration.
  bool meta_stat = vital::algo::video_input::
    check_nested_algo_configuration( "metadata_source", config );

  return image_stat && meta_stat;
}


// ------------------------------------------------------------------
void
video_input_split
::open( std::string name )
{
  if ( ! d->d_image_source )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "invalid video_input algorithm for image source" );
  }
  if ( ! d->d_metadata_source )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "invalid video_input algorithm for metadata source" );
  }
  d->d_image_source->open( name );
  d->d_metadata_source->open( name );

  auto const& is_caps = d->d_image_source->get_implementation_capabilities();
  auto const& ms_caps = d->d_metadata_source->get_implementation_capabilities();

  typedef vital::algo::video_input vi;

  // pass through and combine capabilities
  set_capability( vi::HAS_EOV,
                  is_caps.capability( vi::HAS_EOV) ||
                  ms_caps.capability( vi::HAS_EOV) );
  set_capability( vi::HAS_FRAME_NUMBERS,
                  is_caps.capability( vi::HAS_FRAME_NUMBERS ) ||
                  ms_caps.capability( vi::HAS_FRAME_NUMBERS ) );
  set_capability( vi::HAS_FRAME_DATA,
                  is_caps.capability( vi::HAS_FRAME_DATA ) );
  set_capability( vi::HAS_FRAME_TIME,
                  ms_caps.capability( vi::HAS_FRAME_TIME ) );
  set_capability( vi::HAS_METADATA,
                  ms_caps.capability( vi::HAS_METADATA ) );
  set_capability( vi::HAS_ABSOLUTE_FRAME_TIME,
                  ms_caps.capability( vi::HAS_ABSOLUTE_FRAME_TIME ) );
  d->d_has_timeout = ms_caps.capability( vi::HAS_TIMEOUT ) &&
                     is_caps.capability( vi::HAS_TIMEOUT );
  set_capability( vi::HAS_TIMEOUT, d->d_has_timeout );
  set_capability( vi::IS_SEEKABLE,
                  is_caps.capability( vi::IS_SEEKABLE) &&
                  ms_caps.capability( vi::IS_SEEKABLE) );
}


// ------------------------------------------------------------------
void
video_input_split
::close()
{
  if( d->d_image_source )
  {
    d->d_image_source->close();
  }
  if( d->d_metadata_source )
  {
    d->d_metadata_source->close();
  }
}


// ------------------------------------------------------------------
bool
video_input_split
::end_of_video() const
{
  return (!d->d_image_source || d->d_image_source->end_of_video()) ||
         (!d->d_metadata_source || d->d_metadata_source->end_of_video());
}


// ------------------------------------------------------------------
bool
video_input_split
::good() const
{
  return (d->d_image_source && d->d_image_source->good()) &&
         (d->d_metadata_source && d->d_metadata_source->good());
}

// ------------------------------------------------------------------
bool
video_input_split
::seekable() const
{
  return (d->d_image_source && d->d_image_source->seekable()) &&
         (d->d_metadata_source && d->d_metadata_source->seekable());
}

// ------------------------------------------------------------------
size_t
video_input_split
::num_frames() const
{
  if (d->d_image_source && d->d_metadata_source)
  {
    return std::min(d->d_image_source->num_frames(),
                    d->d_metadata_source->num_frames());
  }
  else
  {
    return 0;
  }
}

// ------------------------------------------------------------------
bool
video_input_split
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout )
{
  // Check for at end of data
  if ( this->end_of_video() )
  {
    return false;
  }
  // if timeout is not supported by both sources
  // then do not pass a time out value to either
  if ( ! d->d_has_timeout )
  {
    timeout = 0;
  }

  kwiver::vital::timestamp image_ts;
  bool image_stat = d->d_image_source->next_frame( image_ts, timeout );

  kwiver::vital::timestamp metadata_ts;
  bool meta_stat = d->d_metadata_source->next_frame( metadata_ts, timeout );

  if( !image_stat || !meta_stat )
  {
    return false;
  }

  // Both timestamps should be the same
  ts = merge_timestamps( image_ts, metadata_ts );

  return true;
} // video_input_split::next_frame

// ------------------------------------------------------------------
bool
video_input_split
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  // if timeout is not supported by both sources
  // then do not pass a time out value to either
  if ( ! d->d_has_timeout )
  {
    timeout = 0;
  }

  kwiver::vital::timestamp image_ts;
  bool image_stat = d->d_image_source->seek_frame( image_ts, frame_number, timeout );

  kwiver::vital::timestamp metadata_ts;
  bool meta_stat = d->d_metadata_source->seek_frame( metadata_ts, frame_number, timeout );

  if( !image_stat || !meta_stat )
  {
    return false;
  }

  // Both timestamps should be the same
  ts = metadata_ts;
  if (image_ts != metadata_ts )
  {
    if ( image_ts.get_frame() == metadata_ts.get_frame() )
    {
      if ( image_ts.has_valid_time() && ! metadata_ts.has_valid_time() )
      {
        ts.set_time_usec( image_ts.get_time_usec() );
      }
      else if ( image_ts.has_valid_time() && metadata_ts.has_valid_time() )
      {
        LOG_WARN( logger(), "Timestamps from image and metadata sources have different time" );
      }
    }
    else
    {
      // throw something?
      LOG_WARN( logger(), "Timestamps from image and metadata sources are out of sync" );
    }
  }

  return true;
} // video_input_split::seek_frame

// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_split
::frame_timestamp() const
{
  // Check for at end of data
  if ( this->end_of_video() )
  {
    return {};
  }

  auto const& image_ts = d->d_image_source->frame_timestamp();
  auto const& metadata_ts = d->d_metadata_source->frame_timestamp();

  return merge_timestamps( image_ts, metadata_ts );
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_split
::frame_image()
{
  return d->d_image_source->frame_image();
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_split
::frame_metadata()
{
  auto md_vec1 = d->d_image_source->frame_metadata();
  auto md_vec2 = d->d_metadata_source->frame_metadata();
  md_vec1.insert( md_vec1.end(), md_vec2.begin(), md_vec2.end() );
  return md_vec1;
}

kwiver::vital::metadata_map_sptr
video_input_split
::metadata_map()
{
  vital::metadata_map::map_metadata_t output_map;

  auto md_map1 = d->d_image_source->metadata_map()->metadata();
  auto md_map2 = d->d_metadata_source->metadata_map()->metadata();
  std::set<kwiver::vital::frame_id_t> merged_keys;
  for ( auto& md : md_map1 )
  {
    if ( md_map2.find(md.first) != md_map2.end() )
    {
      md.second.insert( md.second.end(),
                        md_map2[md.first].begin(),
                        md_map2[md.first].end() );
      merged_keys.insert(md.first);
    }
  }
  for (auto md : md_map2 )
  {
    if ( md_map1.find(md.first) != md_map1.end() &&
         merged_keys.find(md.first) == merged_keys.end() )
    {
      md_map1[md.first].insert( md_map1[md.first].end(),
                                md.second.begin(),
                                md.second.end() );
    }
  }

  return std::make_shared<kwiver::vital::simple_metadata_map>(md_map1);
}


// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_split
::merge_timestamps(
    kwiver::vital::timestamp const& image_ts,
    kwiver::vital::timestamp const& metadata_ts ) const
{
  auto ts = metadata_ts;
  if ( image_ts != metadata_ts )
  {
    if ( image_ts.get_frame() == metadata_ts.get_frame() )
    {
      if ( image_ts.has_valid_time() && metadata_ts.has_valid_time() )
      {
        LOG_WARN( logger(), "Timestamps from image and metadata sources have different time" );
      }
      else if ( image_ts.has_valid_time() )
      {
        ts.set_time_usec( image_ts.get_time_usec() );
      }
      else if ( metadata_ts.has_valid_time() )
      {
        ts.set_time_usec( metadata_ts.get_time_usec() );
      }
    }
    else
    {
      // throw something?
      LOG_WARN( logger(), "Timestamps from image and metadata sources are out of sync" );
    }
  }

  return ts;
}

} } }     // end namespace
