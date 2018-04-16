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


namespace kwiver {
namespace arrows {
namespace core {

class video_input_splice::priv
{
public:
  priv() :
  d_has_timeout(false),
  d_is_seekable(false)
  { }

  bool d_has_timeout;
  bool d_is_seekable;

  // Vector of video sources
  std::vector< vital::algo::video_input_sptr > d_video_sources;

  // Pointer to the active source
  vital::algo::video_input_sptr d_active_source;
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

  // Names of the other video sources - TODO

  // Configure the dependent video sources - TODO

  return config;
}


// ------------------------------------------------------------------
void
video_input_splice
::set_configuration( vital::config_block_sptr config )
{
  // Set the configs for the other video sources - TODO
}


// ------------------------------------------------------------------
bool
video_input_splice
::check_configuration( vital::config_block_sptr config ) const
{
  bool retVal = true;
  // Check the configurations. - TODO

  return retVal;
}


// ------------------------------------------------------------------
void
video_input_splice
::open( std::string name )
{
  // Loop through and open them - TODO
  // Do we open them all at once or just when needed.

  // Set capability if it is common to all sources.
  // set_capability( vi::HAS_EOV,
  //                 is_caps.capability( vi::HAS_EOV) ||
  //                 ms_caps.capability( vi::HAS_EOV) );
  // set_capability( vi::HAS_FRAME_NUMBERS,
  //                 is_caps.capability( vi::HAS_FRAME_NUMBERS ) ||
  //                 ms_caps.capability( vi::HAS_FRAME_NUMBERS ) );
  // set_capability( vi::HAS_FRAME_DATA,
  //                 is_caps.capability( vi::HAS_FRAME_DATA ) );
  // set_capability( vi::HAS_FRAME_TIME,
  //                 ms_caps.capability( vi::HAS_FRAME_TIME ) );
  // set_capability( vi::HAS_METADATA,
  //                 ms_caps.capability( vi::HAS_METADATA ) );
  // set_capability( vi::HAS_ABSOLUTE_FRAME_TIME,
  //                 ms_caps.capability( vi::HAS_ABSOLUTE_FRAME_TIME ) );
  // d->d_has_timeout = ms_caps.capability( vi::HAS_TIMEOUT ) &&
  //                    is_caps.capability( vi::HAS_TIMEOUT );
  // set_capability( vi::HAS_TIMEOUT, d->d_has_timeout );
  // set_capability( vi::IS_SEEKABLE,
  //                 is_caps.capability( vi::IS_SEEKABLE) &&
  //                 ms_caps.capability( vi::IS_SEEKABLE) );
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
}


// ------------------------------------------------------------------
bool
video_input_splice
::end_of_video() const
{
  // TODO: add logic to see if we are on the last source and it is at its end.
  return true;
}


// ------------------------------------------------------------------
bool
video_input_splice
::good() const
{
  if ( d->d_active_source )
  {
    return d->d_active_source->good();
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
  // TODO: Pre-calculate?
  return 0;
}

// ------------------------------------------------------------------
bool
video_input_splice
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout )
{
  // TODO: lots!
  kwiver::vital::timestamp image_ts;

  return true;
} // video_input_splice::next_frame

// ------------------------------------------------------------------
bool
video_input_splice
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  // if timeout is not supported by all sources
  // then do not pass a time out value to active source
  if ( ! d->d_has_timeout )
  {
    timeout = 0;
  }

  kwiver::vital::timestamp image_ts;
  // TODO: a lot!!

  return true;
} // video_input_splice::seek_frame

// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_splice
::frame_timestamp() const
{
  return d->d_active_source->frame_timestamp();
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_splice
::frame_image()
{
  return d->d_active_source->frame_image();
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_splice
::frame_metadata()
{
  return d->d_active_source->frame_metadata();
}

kwiver::vital::metadata_map_sptr
video_input_splice
::metadata_map()
{
  // TODO: pre-calculate
  return d->d_active_source->metadata_map();
}

} } }     // end namespace
