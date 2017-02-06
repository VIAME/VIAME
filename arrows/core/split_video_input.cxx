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

#include "split_video_input.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>
#include <vital/types/geo_lat_lon.h>
#include <vital/algo/image_io.h>

#include <vital/video_metadata/video_metadata.h>
#include <vital/video_metadata/video_metadata_traits.h>

#include <kwiversys/SystemTools.hxx>

#include <vector>
#include <stdint.h>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace core {

class split_video_input::priv
{
public:
  priv()
    : c_start_at_frame( 0 )
    , c_stop_after_frame( 0 )
    , c_frame_time( 0.03333 )
    , d_at_eov( false )
  { }

  // Configuration values
  unsigned int c_start_at_frame;
  unsigned int c_stop_after_frame;
  float c_frame_time;
  std::string c_image_reader;
  std::string c_metadata_reader;

  // local state
  bool d_at_eov;

  std::vector < kwiver::vital::path_t > d_metadata_files;
  std::vector < kwiver::vital::path_t >::const_iterator d_current_file;
  kwiver::vital::timestamp::frame_t d_frame_number;
  kwiver::vital::timestamp::time_t d_frame_time;

  vital::video_metadata_sptr d_metadata;

  // processing classes
  vital::algo::video_input_sptr d_image_reader;
  vital::algo::video_input_sptr d_metadata_reader;

};


// ------------------------------------------------------------------
split_video_input
::split_video_input()
  : d( new split_video_input::priv )
{
  attach_logger( "split_video_input" );
}


// ------------------------------------------------------------------
split_video_input
::~split_video_input()
{
}


// ------------------------------------------------------------------
split_video_input
::split_video_input( split_video_input const& other )
  : d( new split_video_input::priv )
{
  // copy CTOR

  // TBD
}


// ------------------------------------------------------------------
vital::config_block_sptr
split_video_input
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "start_at_frame", d->c_start_at_frame,
                     "Frame number (from 1) to start processing video input. "
                     "If set to zero, start at the beginning of the video." );

  config->set_value( "stop_after_frame", d->c_stop_after_frame,
                     "Number of frames to supply. If set to zero then supply all frames after start frame." );

  config->set_value( "frame_time", d->c_frame_time, "Inter frame time in seconds. "
                     "The generated timestamps will have the specified number of seconds in the generated "
                     "timestamps for sequential frames. This can be used to simulate a frame rate in a "
                     "video stream application.");

  config->set_value( "image_reader", "",
                     "Config block that configures the image reader. image_reader:type specifies the "
                     "implementation type. Other configuration items may be needed, depending on the implementation.");

  config->set_value( "metadata_reader", "",
                     "Config block that configures the metadata reader. metadata_reader:type specifies the "
                     "implementation type. Other configuration items may be needed, depending on the implementation.");

  return config;
}


// ------------------------------------------------------------------
void
split_video_input
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d->c_start_at_frame = config->get_value<vital::timestamp::frame_t>(
    "start_at_frame", d->c_start_at_frame );

  d->c_stop_after_frame = config->get_value<vital::timestamp::frame_t>(
    "stop_after_frame", d->c_stop_after_frame );

  // get frame time
  d->c_frame_time = config->get_value<float>(
    "frame_time", d->c_frame_time );

  // Setup actual reader algorithm
  vital::algo::video_input::set_nested_algo_configuration( "image_reader", config, d->d_image_reader);
  if ( ! d->d_image_reader )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "unable to create image_reader." );
  }

  vital::algo::video_input::set_nested_algo_configuration( "metadata_reader", config, d->d_metadata_reader);
  if ( ! d->d_metadata_reader )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "unable to create metadata_reader." );
  }
}


// ------------------------------------------------------------------
bool
split_video_input
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the image reader configuration.
  bool image_stat = vital::algo::image_io::check_nested_algo_configuration( "image_reader", config );

  // Check the metadata reader configuration.
  bool meta_stat = vital::algo::image_io::check_nested_algo_configuration( "metadata_reader", config );

  return image_stat && meta_stat;
}


// ------------------------------------------------------------------
void
split_video_input
::open( std::string name )
{
  // open file and read lines

  //+ how to initialize two readers with one name???

}


// ------------------------------------------------------------------
void
split_video_input
::close()
{
}


// ------------------------------------------------------------------
bool
split_video_input
::end_of_video() const
{
  return d->d_at_eov;
}


// ------------------------------------------------------------------
bool
split_video_input
::good() const
{
  // This could use a more nuanced approach
  return true;
}


// ------------------------------------------------------------------
bool
split_video_input
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout ) // not supported
{
  // Check for at end of data
  if ( d->d_at_eov )
  {
    return false;
  }

  kwiver::vital::timestamp image_ts;
  bool image_stat = d->d_image_reader->next_frame( image_ts, timeout );

  kwiver::vital::timestamp metadata_ts;
  bool meta_stat = d->d_metadata_reader->next_frame( metadata_ts, timeout );

  // Both timestamps should be the same
  if (image_ts != metadata_ts )
  {
    // throw something.
  }

  return image_stat && meta_stat;;
} // split_video_input::next_frame


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
split_video_input
::frame_image()
{
  return d->d_image_reader->frame_image();
}


// ------------------------------------------------------------------
kwiver::vital::video_metadata_vector
split_video_input
::frame_metadata()
{
  return d->d_metadata_reader->frame_metadata();
}

} } }     // end namespace
