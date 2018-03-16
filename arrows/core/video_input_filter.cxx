/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include "video_input_filter.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/exceptions.h>

#include <vector>

namespace kwiver {
namespace arrows {
namespace core {

class video_input_filter::priv
{
public:
  priv()
    : c_start_at_frame( 0 )
    , c_stop_after_frame( 0 )
    , c_frame_rate( 30.0 )
    , d_at_eov( false )
  { }

  // Configuration values
  vital::frame_id_t c_start_at_frame;
  vital::frame_id_t c_stop_after_frame;
  double c_frame_rate;

  // local state
  bool d_at_eov;

  // processing classes
  vital::algo::video_input_sptr d_video_input;
};


// ------------------------------------------------------------------
video_input_filter
::video_input_filter()
  : d( new video_input_filter::priv )
{
  attach_logger( "video_input_filter" );
}


// ------------------------------------------------------------------
video_input_filter
::~video_input_filter()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
video_input_filter
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "start_at_frame", d->c_start_at_frame,
                     "Frame number (from 1) to start processing video input. "
                     "If set to zero, start at the beginning of the video." );

  config->set_value( "stop_after_frame", d->c_stop_after_frame,
                     "End the video after passing this frame number. "
                     "Set this value to 0 to disable filter.");

  config->set_value( "frame_rate", d->c_frame_rate, "Number of frames per second. "
                     "If the video does not provide a valid time, use this rate "
                     "to compute frame time.  Set 0 to disable.");

  vital::algo::video_input::
    get_nested_algo_configuration( "video_input", config, d->d_video_input );

  return config;
}


// ------------------------------------------------------------------
void
video_input_filter
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d->c_start_at_frame = config->get_value<vital::frame_id_t>(
    "start_at_frame", d->c_start_at_frame );

  d->c_stop_after_frame = config->get_value<vital::frame_id_t>(
    "stop_after_frame", d->c_stop_after_frame );

  // get frame time
  d->c_frame_rate = config->get_value<double>(
    "frame_rate", d->c_frame_rate );

  // Setup actual video input algorithm
  vital::algo::video_input::
    set_nested_algo_configuration( "video_input", config, d->d_video_input);
 }


// ------------------------------------------------------------------
bool
video_input_filter
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the video input configuration.
  return vital::algo::video_input::check_nested_algo_configuration( "video_input", config );
}


// ------------------------------------------------------------------
void
video_input_filter
::open( std::string name )
{
  if ( ! d->d_video_input )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "invalid video_input." );
  }
  d->d_video_input->open( name );
  d->d_at_eov = false;

  auto const& vi_caps = d->d_video_input->get_implementation_capabilities();

  typedef vital::algo::video_input vi;

  // pass through capabilities, modified as needed
  set_capability( vi::HAS_EOV,
                  vi_caps.capability( vi::HAS_EOV) ||
                  d->c_stop_after_frame > 0 );
  set_capability( vi::HAS_FRAME_NUMBERS,
                  vi_caps.capability( vi::HAS_FRAME_NUMBERS ) );
  set_capability( vi::HAS_FRAME_DATA,
                  vi_caps.capability( vi::HAS_FRAME_DATA ) );
  set_capability( vi::HAS_FRAME_TIME,
                  vi_caps.capability( vi::HAS_FRAME_TIME ) ||
                  d->c_frame_rate > 0 );
  set_capability( vi::HAS_METADATA,
                  vi_caps.capability( vi::HAS_METADATA ) );
  set_capability( vi::HAS_ABSOLUTE_FRAME_TIME,
                  vi_caps.capability( vi::HAS_ABSOLUTE_FRAME_TIME ) );
  set_capability( vi::HAS_TIMEOUT,
                  vi_caps.capability( vi::HAS_TIMEOUT ) );

}


// ------------------------------------------------------------------
void
video_input_filter
::close()
{
  if( d->d_video_input )
  {
    d->d_video_input->close();
  }
}


// ------------------------------------------------------------------
bool
video_input_filter
::end_of_video() const
{
  return d->d_at_eov;
}


// ------------------------------------------------------------------
bool
video_input_filter
::good() const
{
  if( ! d->d_video_input )
  {
    return false;
  }
  return d->d_video_input->good();
}


// ------------------------------------------------------------------
bool
video_input_filter
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout ) // not supported
{
  // Check for at end of data
  if( d->d_at_eov )
  {
    return false;
  }

  bool status = d->d_video_input->next_frame( ts, timeout );
  // step through additional frames to reach the start frame
  while( status && ts.get_frame() < d->c_start_at_frame)
  {
    status = d->d_video_input->next_frame( ts, timeout );
  }

  if( d->c_stop_after_frame > 0 &&
      ts.get_frame() > d->c_stop_after_frame )
  {
    d->d_at_eov = true;
    return false;
  }

  // set the frame time base on rate if missing
  if( d->c_frame_rate > 0 && !ts.has_valid_time() )
  {
    ts.set_time_seconds( ts.get_frame() / d->c_frame_rate );
  }

  return status;
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_filter
::frame_image()
{
  if( ! this->end_of_video() )
  {
    return d->d_video_input->frame_image();
  }
  return nullptr;
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_filter
::frame_metadata()
{
  if( ! this->end_of_video() )
  {
    return d->d_video_input->frame_metadata();
  }
  return kwiver::vital::metadata_vector();
}

} } }     // end namespace
