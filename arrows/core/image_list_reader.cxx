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

#include "image_list_reader.h"

#include <vital/algorithm_plugin_manager.h>
#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/algo/image_io.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>

#include <kwiversys/SystemTools.hxx>

#include <vector>
#include <stdint.h>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace core {

class image_list_reader::priv
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
  std::vector< std::string > c_search_path;
  float c_frame_time;

  // local state
  bool d_at_eov;

  std::vector < kwiver::vital::path_t > m_files;
  std::vector < kwiver::vital::path_t >::const_iterator m_current_file;
  kwiver::vital::timestamp::frame_t m_frame_number;
  kwiver::vital::timestamp::time_t m_frame_time;
  kwiver::vital::image_container_sptr m_image;

  // processing classes
  vital::algo::image_io_sptr m_image_reader;
};


// ------------------------------------------------------------------
image_list_reader
::image_list_reader()
  : d( new image_list_reader::priv )
{
  attach_logger( "image_list_reader" );

  set_capability( vital::algo::video_input::HAS_EOV, true );
  set_capability( vital::algo::video_input::HAS_FRAME_NUMBERS, true );
  set_capability( vital::algo::video_input::HAS_FRAME_TIME, true );
  set_capability( vital::algo::video_input::HAS_FRAME_DATA, true );

  set_capability( vital::algo::video_input::HAS_METADATA, false );
  set_capability( vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME, false );
  set_capability( vital::algo::video_input::HAS_TIMEOUT, false );
}


// ------------------------------------------------------------------
image_list_reader
::~image_list_reader()
{
}


// ------------------------------------------------------------------
image_list_reader
::image_list_reader( image_list_reader const& other )
{
  // copy CTOR

  // TBD
}


// ------------------------------------------------------------------
vital::config_block_sptr
image_list_reader
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

  config->set_value( "path", "",
                     "Path to search for image file. The format is the same as the standard "
                     "path specification, a set of directories separated by a colon (':')" );

  config->set_value( "image_reader", "",
                     "Config block that configures the image reader. image_reader:type specifies the "
                     "implementation type. Other configuration items may be needed depending on the implementation.");

  return config;
}


// ------------------------------------------------------------------
void
image_list_reader
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  {
    config->merge_config(in_config);
  }

  d->c_start_at_frame = config->get_value<vital::timestamp::frame_t>(
    "start_at_frame", d->c_start_at_frame );

  d->c_stop_after_frame = config->get_value<vital::timestamp::frame_t>(
    "stop_after_frame", d->c_stop_after_frame );

  // Extract string and create vector of directories
  std::string path = config->get_value<std::string>( "path", "" );
  kwiver::vital::tokenize( path, d->c_search_path, ":", true );
  d->c_search_path.push_back( "." ); // add current directory

  // get frame time
  d->c_frame_time = config->get_value<float>(
    "frame_time", d->c_frame_time );

  // Setup actual reader algorithm
  vital::algo::image_io::set_nested_algo_configuration( "image_reader", config, d->m_image_reader);
  if ( ! d->m_image_reader )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "unable to create image_reader." );
  }
}


// ------------------------------------------------------------------
bool
image_list_reader
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the reader configuration.
  if ( ! vital::algo::image_io::check_nested_algo_configuration( "image_reader", config ) )
  {
    return false;
  }
  return true;
}


// ------------------------------------------------------------------
void
image_list_reader
::open( std::string list_name )
{
  // open file and read lines
  std::ifstream ifs( list_name.c_str() );
  if ( ! ifs )
  {
    std::stringstream msg;
    msg <<  "Could not open image list \"" << list_name << "\"";
    throw kwiver::vital::invalid_file( list_name, "Could not open file" );
  }

  kwiver::vital::data_stream_reader stream_reader( ifs );

  // verify and get file names in a list
  for ( std::string line; stream_reader.getline( line ); /* null */ )
  {
    std::string resolved_file = line;
    if ( ! kwiversys::SystemTools::FileExists( line ) )
    {
      // Resolve against specified path
      resolved_file = kwiversys::SystemTools::FindFile( line, d->c_search_path, true );
      if ( resolved_file.empty() )
      {
        throw kwiver::vital::file_not_found_exception( line, "could not locate file in path" );
      }
    }

    d->m_files.push_back( resolved_file );
  } // end for

  d->m_current_file = d->m_files.begin();
  d->m_frame_number = 1;
}


// ------------------------------------------------------------------
void
image_list_reader
::close()
{
  // Nothing to do here
}


// ------------------------------------------------------------------
bool
image_list_reader
::end_of_video() const
{
  return ( d->m_current_file == d->m_files.end() );
}


// ------------------------------------------------------------------
bool
image_list_reader
::good() const
{
  // This could use a more nuanced approach
  return true;
}


// ------------------------------------------------------------------
bool
image_list_reader
::next_frame( kwiver::vital::timestamp& ts,
              uint32_t                  timeout )
{
  // returns timestamp
  // does not support timeout
  if ( d->m_current_file == d->m_files.end() )
  {
    return false;
  }

  // still have an image to read
  std::string a_file = *d->m_current_file;

  LOG_DEBUG( m_logger, "reading image from file \"" << a_file << "\"" );

  // read image file
  //
  // This call returns a *new* image container. This is good since
  // we are going to pass it downstream using the sptr.
  d->m_image = d->m_image_reader->load( a_file );

  // --- debug
#if defined DEBUG
  cv::Mat image = algorithms::ocv::image_container::vital_to_ocv( d->m_image->get_image() );
  namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
  imshow( "Display window", image );                   // Show our image inside it.

  waitKey(0);                 // Wait for a keystroke in the window
#endif
  // -- end debug

  // Return timestamp
  ts = kwiver::vital::timestamp( d->m_frame_time, d->m_frame_number );

  // update timestamp
  ++d->m_frame_number;
  d->m_frame_time += d->c_frame_time;

  ++d->m_current_file;

  return true;
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
image_list_reader
::frame_image()
{
  return d->m_image;
}


// ------------------------------------------------------------------
kwiver::vital::video_metadata_vector
image_list_reader
::frame_metadata()
{
  // There is no metadata available at this time.  Capability shows
  // that there are no metadata, so this should not be called.  Return
  // empty vector.
  return kwiver::vital::video_metadata_vector();
}

} } }     // end namespace
