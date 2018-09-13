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

#include "video_input_image_list.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/types/metadata_traits.h>
#include <vital/algo/image_io.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>

#include <kwiversys/SystemTools.hxx>

#include <string>
#include <vector>
#include <stdint.h>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace core {

class video_input_image_list::priv
{
public:
  priv()
  : m_current_file( m_files.end() )
  , m_frame_number( 0 )
  , m_image( nullptr )
  , m_have_metadata_map( false )
  {}

  // Configuration values
  std::vector< std::string > c_search_path;

  // local state
  std::vector < kwiver::vital::path_t > m_files;
  std::vector < kwiver::vital::path_t >::const_iterator m_current_file;
  kwiver::vital::frame_id_t m_frame_number;
  kwiver::vital::image_container_sptr m_image;

  // metadata map
  bool m_have_metadata_map;
  vital::metadata_map::map_metadata_t m_metadata_map;

  // processing classes
  vital::algo::image_io_sptr m_image_reader;
};


// ------------------------------------------------------------------
video_input_image_list
::video_input_image_list()
  : d( new video_input_image_list::priv )
{
  attach_logger( "arrows.core.video_input_image_list" );

  set_capability( vital::algo::video_input::HAS_EOV, true );
  set_capability( vital::algo::video_input::HAS_FRAME_NUMBERS, true );
  set_capability( vital::algo::video_input::HAS_FRAME_DATA, true );
  set_capability( vital::algo::video_input::HAS_METADATA, true );

  set_capability( vital::algo::video_input::HAS_FRAME_TIME, false );
  set_capability( vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME, false );
  set_capability( vital::algo::video_input::HAS_TIMEOUT, false );
  set_capability( vital::algo::video_input::IS_SEEKABLE, true );
}


// ------------------------------------------------------------------
video_input_image_list
::~video_input_image_list()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
video_input_image_list
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "path", "",
                     "Path to search for image file. "
                     "If a file name is not absolute, this list of directories is scanned to find the file. "
                     "The current directory '.' is automatically appended to the end of the path. "
                     "The format of this path is the same as the standard "
                     "path specification, a set of directories separated by a colon (':')" );

  vital::algo::image_io::
    get_nested_algo_configuration( "image_reader", config, d->m_image_reader );

  return config;
}


// ------------------------------------------------------------------
void
video_input_image_list
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Extract string and create vector of directories
  std::string path = config->get_value<std::string>( "path", "" );
  kwiver::vital::tokenize( path, d->c_search_path, ":", kwiver::vital::TokenizeTrimEmpty );
  d->c_search_path.push_back( "." ); // add current directory

  // Setup actual reader algorithm
  vital::algo::image_io::
    set_nested_algo_configuration( "image_reader", config, d->m_image_reader );
}


// ------------------------------------------------------------------
bool
video_input_image_list
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the reader configuration.
  return vital::algo::image_io::
    check_nested_algo_configuration( "image_reader", config );
}


// ------------------------------------------------------------------
void
video_input_image_list
::open( std::string list_name )
{
  typedef kwiversys::SystemTools ST;

  // close the video in case already open
  this->close();

  // open file and read lines
  std::ifstream ifs( list_name.c_str() );
  if ( ! ifs )
  {
    throw kwiver::vital::invalid_file( list_name, "Could not open file" );
  }
  if ( ! d->m_image_reader )
  {
    throw kwiver::vital::algorithm_configuration_exception( type_name(), impl_name(),
          "invalid image_reader." );
  }

  // Add directory that contains the list file to the path
  std::string list_path = ST::GetFilenamePath( list_name );
  if ( ! list_path.empty() )
  {
    d->c_search_path.push_back( list_path );
  }

  kwiver::vital::data_stream_reader stream_reader( ifs );

  // verify and get file names in a list
  std::string data_dir = "";
  std::string line;
  // Read the first line and determine to file location
  if ( stream_reader.getline( line ) )
  {
    std::string resolved_file = line;
    if ( ! ST::FileExists( resolved_file ) )
    {
      // Resolve against specified path
      resolved_file = ST::FindFile( line, d->c_search_path, true );
      if ( resolved_file.empty() )
      {
        throw kwiver::vital::
          file_not_found_exception( line, "could not locate file in path" );
      }
      if( ST::StringEndsWith( resolved_file.c_str(), line.c_str() ) )
      {
        // extract the prefix added to get the full path
        data_dir = resolved_file.substr(0, resolved_file.size() - line.size() );
      }
    }
    d->m_files.push_back( resolved_file );
  }
  // Read the rest of the file and validate paths
  // Only check the same data_dir used to resolve the first frame
  while ( stream_reader.getline( line ) )
  {
    std::string resolved_file = line;
    if ( ! ST::FileExists( resolved_file ) )
    {
      resolved_file = data_dir + line;
      if ( ! ST::FileExists( resolved_file ) )
      {
        throw kwiver::vital::
          file_not_found_exception( line,
              "could not locate file relative to \"" + data_dir + "\"" );
      }
    }

    d->m_files.push_back( resolved_file );
  } // end while

  d->m_current_file = d->m_files.begin();
  d->m_frame_number = 0;
}


// ------------------------------------------------------------------
void
video_input_image_list
::close()
{
  d->m_files.clear();
  d->m_current_file = d->m_files.end();
  d->m_frame_number = 0;
  d->m_image = nullptr;
}


// ------------------------------------------------------------------
bool
video_input_image_list
::end_of_video() const
{
  return ( d->m_current_file == d->m_files.end() );
}


// ------------------------------------------------------------------
bool
video_input_image_list
::good() const
{
  return d->m_frame_number > 0 && ! this->end_of_video();
}

// ------------------------------------------------------------------
bool
video_input_image_list
::seekable() const
{
  return true;
}

// ------------------------------------------------------------------
size_t
video_input_image_list
::num_frames() const
{
  return d->m_files.size();
}

// ------------------------------------------------------------------
bool
video_input_image_list
::next_frame( kwiver::vital::timestamp& ts,
              uint32_t                  timeout )
{
  // returns timestamp
  // does not support timeout
  if ( this->end_of_video() )
  {
    return false;
  }

  // clear the last loaded image
  d->m_image = nullptr;

  // If this is the first call to next_frame() then
  // do not increment the file iteration.
  // next_frame() must be called once before accessing the first frame.
  if ( d->m_frame_number > 0 )
  {
    ++d->m_current_file;
  }

  ++d->m_frame_number;

  // Return timestamp
  ts = this->frame_timestamp();

  return ! this->end_of_video();
}

// ------------------------------------------------------------------
bool
video_input_image_list
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  // Check if requested frame exists
  if (frame_number > static_cast<int>( d->m_files.size() ) || frame_number <= 0)
  {
    return false;
  }

  // Adjust frame number if this is the first call to seek_frame or next_frame
  if (d->m_frame_number == 0)
  {
    d->m_frame_number = 1;
  }

  // Calculate distance to new frame
  kwiver::vital::timestamp::frame_t frame_diff =
    frame_number - d->m_frame_number;
  d->m_current_file += frame_diff;
  d->m_frame_number = frame_number;

  // clear the last loaded image
  d->m_image = nullptr;

  // Return timestamp
  ts = kwiver::vital::timestamp();

  ts.set_frame( d->m_frame_number );

  return ! this->end_of_video();
}

// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_image_list
::frame_timestamp() const
{
  if ( this->end_of_video() )
  {
    return {};
  }

  kwiver::vital::timestamp ts;

  ts.set_frame( d->m_frame_number );

  return ts;
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_image_list
::frame_image()
{
  if ( !d->m_image && this->good() )
  {
    LOG_DEBUG( logger(), "reading image from file \"" << *d->m_current_file << "\"" );

    // read image file
    //
    // This call returns a *new* image container. This is good since
    // we are going to pass it downstream using the sptr.
    d->m_image = d->m_image_reader->load( *d->m_current_file );
  }
  return d->m_image;
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_image_list
::frame_metadata()
{
  if ( ! this->good() )
  {
    return vital::metadata_vector();
  }
  // For now, the only metadata is the filename of the image
  auto md = std::make_shared<vital::metadata>();
  md->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_FILENAME,
                              *d->m_current_file ) );
  vital::metadata_vector mdv(1, md);
  return mdv;
}

kwiver::vital::metadata_map_sptr
video_input_image_list
::metadata_map()
{
  if ( !d->m_have_metadata_map )
  {
    kwiver::vital::timestamp::frame_t fn = 0;
    for (const auto& f: d->m_files)
    {
      ++fn;
      // For now, the only metadata is the filename
      auto md = std::make_shared<vital::metadata>();
      md->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_FILENAME, f ) );
      vital::metadata_vector mdv(1, md);
      d->m_metadata_map[fn] = mdv;
    }

    d->m_have_metadata_map = true;
  }

  return std::make_shared<kwiver::vital::simple_metadata_map>(d->m_metadata_map);
}

} } }     // end namespace
