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

#include "video_input_pos.h"

#include <vital/vital_types.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/timestamp.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>

#include <vital/io/metadata_io.h>

#include <kwiversys/SystemTools.hxx>

#include <fstream>


namespace kwiver {
namespace arrows {
namespace core {

class video_input_pos::priv
{
public:
  priv()
  : c_meta_extension( ".pos" )
  , d_current_files( d_img_md_files.end() )
  , d_frame_number( 0 )
  , d_metadata( nullptr )
  , d_have_metadata_map( false )
  {}

  // Configuration values
  std::string c_meta_directory;
  std::string c_meta_extension;
  std::string c_image_list_file;

  // local state
  typedef std::pair < vital::path_t, vital::path_t > path_pair_t;
  std::vector < path_pair_t > d_img_md_files;
  std::vector < path_pair_t >::const_iterator d_current_files;
  kwiver::vital::frame_id_t d_frame_number;

  vital::metadata_sptr d_metadata;

  // metadata map
  bool d_have_metadata_map;
  vital::metadata_map::map_metadata_t d_metadata_map;

  // Read and process metadata from file
  vital::metadata_sptr process_metadata(path_pair_t paths,
                        vital::timestamp::frame_t frame,
                        vital::timestamp& ts)
  {
    vital::metadata_sptr metadata = nullptr;

    if ( ! paths.second.empty() )
    {
      // Open next file in the list
      metadata = vital::read_pos_file( paths.second );
    }

    // Include the path to the image
    if ( metadata )
    {
      metadata->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_FILENAME,
                                        paths.first) );
    }

    // Return timestamp
    ts.set_frame( frame );
    if ( metadata )
    {
      if ( metadata->has( vital::VITAL_META_GPS_SEC ) )
      {
        double gps_sec = metadata->find( vital::VITAL_META_GPS_SEC ).as_double();
        // TODO: also use gps_week and convert to UTC to get abosolute time
        // or subtract off first frame time to get time relative to start
        ts.set_time_seconds( gps_sec );
      }
      metadata->set_timestamp( ts );
    }

    return metadata;
  }
};


// ------------------------------------------------------------------
video_input_pos
::video_input_pos()
  : d( new video_input_pos::priv )
{
  attach_logger( "arrows.core.video_input_pos" );

  set_capability( vital::algo::video_input::HAS_EOV, true );
  set_capability( vital::algo::video_input::HAS_FRAME_NUMBERS, true );
  set_capability( vital::algo::video_input::HAS_FRAME_TIME, true );
  set_capability( vital::algo::video_input::HAS_METADATA, true );

  set_capability( vital::algo::video_input::HAS_FRAME_DATA, false );
  set_capability( vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME, false ); // MAYBE
  set_capability( vital::algo::video_input::HAS_TIMEOUT, false );
  set_capability( vital::algo::video_input::IS_SEEKABLE, true );
}


// ------------------------------------------------------------------
video_input_pos
::~video_input_pos()
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
video_input_pos
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "metadata_directory", d->c_meta_directory,
                     "Name of directory containing metadata files." );

  config->set_value( "metadata_extension", d->c_meta_extension,
                     "File extension of metadata files." );

  return config;
}


// ------------------------------------------------------------------
void
video_input_pos
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d->c_meta_directory = config->get_value<std::string>(
    "metadata_directory", d->c_meta_directory );

  d->c_meta_extension = config->get_value<std::string>(
    "metadata_extension", d->c_meta_extension );
}


// ------------------------------------------------------------------
bool
video_input_pos
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
void
video_input_pos
::open( std::string image_list_name )
{
  typedef kwiversys::SystemTools ST;

  // close the video in case already open
  this->close();

  // open file and read lines
  std::ifstream ifs( image_list_name.c_str() );
  if ( ! ifs )
  {
    throw kwiver::vital::invalid_file( image_list_name, "Could not open file" );
  }

  kwiver::vital::data_stream_reader stream_reader( ifs );

  // verify and get file names in a list
  std::string line;
  while ( stream_reader.getline( line ) )
  {
    // Get base name from file
    std::string resolved_file = d->c_meta_directory;
    resolved_file += "/" + ST::GetFilenameWithoutLastExtension( line )
                     + d->c_meta_extension;
    if ( ! ST::FileExists( resolved_file ) )
    {
      LOG_DEBUG( logger(), "Could not find file " << resolved_file
                 <<". This frame will not have any metadata." );
      resolved_file.clear(); // indicate that the metadata file could not be found
    }

    d->d_img_md_files.push_back( std::make_pair(line, resolved_file) );
  } // end while

  d->d_current_files = d->d_img_md_files.begin();
  d->d_frame_number = 0;
}


// ------------------------------------------------------------------
void
video_input_pos
::close()
{
  d->d_img_md_files.clear();
  d->d_current_files = d->d_img_md_files.end();
  d->d_frame_number = 0;
  d->d_metadata = nullptr;
}


// ------------------------------------------------------------------
bool
video_input_pos
::end_of_video() const
{
  return  d->d_current_files == d->d_img_md_files.end();
}


// ------------------------------------------------------------------
bool
video_input_pos
::good() const
{
  return d->d_frame_number > 0 && ! this->end_of_video();
}

// ------------------------------------------------------------------
bool
video_input_pos
::seekable() const
{
  return true;
}

// ------------------------------------------------------------------
size_t
video_input_pos
::num_frames() const
{
  return d->d_img_md_files.size();
}

// ------------------------------------------------------------------
bool
video_input_pos
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout ) // not supported
{
  // reset current metadata packet and timestamp
  d->d_metadata = nullptr;
  ts = kwiver::vital::timestamp();

  // Check for at end of video
  if ( this->end_of_video() )
  {
    return false;
  }

  // do not increment the iterator on the first call to next_frame()
  if ( d->d_frame_number > 0 )
  {
    ++d->d_current_files;
  }
  ++d->d_frame_number;

  // Check for at end of video
  if ( this->end_of_video() )
  {
    return false;
  }

  if ( ! d->d_current_files->second.empty() )
  {
    // Open next file in the list
    d->d_metadata = vital::read_pos_file( d->d_current_files->second );
  }

  // Return timestamp
  ts = this->frame_timestamp();

  // Include the path to the image
  if ( d->d_metadata )
  {
    d->d_metadata->set_timestamp( ts );
    d->d_metadata->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_FILENAME,
                                           d->d_current_files->first ) );
  }

  return true;
}

// ------------------------------------------------------------------
bool
video_input_pos
::seek_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              kwiver::vital::timestamp::frame_t frame_number,
              uint32_t                  timeout )
{
  // reset current metadata packet and timestamp
  d->d_metadata = nullptr;
  ts = kwiver::vital::timestamp();

  // Check if requested frame exists
  if (frame_number > static_cast<int>( d->d_img_md_files.size() ) || frame_number <= 0)
  {
    return false;
  }

  // Adjust frame number if this is the first call to seek_frame or next_frame
  if (d->d_frame_number == 0)
  {
    d->d_frame_number = 1;
  }

  // Calculate distance to new frame
  kwiver::vital::timestamp::frame_t frame_diff =
    frame_number - d->d_frame_number;
  d->d_current_files += frame_diff;
  d->d_frame_number = frame_number;

  if ( ! d->d_current_files->second.empty() )
  {
    // Open next file in the list
    d->d_metadata = vital::read_pos_file( d->d_current_files->second );
  }

  // Return timestamp
  ts = this->frame_timestamp();

  // Include the path to the image
  if ( d->d_metadata )
  {
    d->d_metadata->set_timestamp( ts );
    d->d_metadata->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_FILENAME,
                                           d->d_current_files->first ) );
  }

  return true;
}


// ------------------------------------------------------------------
kwiver::vital::timestamp
video_input_pos
::frame_timestamp() const
{
  // Check for at end of video
  if ( this->end_of_video() )
  {
    return {};
  }

  kwiver::vital::timestamp ts;

  ts.set_frame( d->d_frame_number );
  if ( d->d_metadata )
  {
    if ( d->d_metadata->has( vital::VITAL_META_GPS_SEC ) )
    {
      double gps_sec = d->d_metadata->find( vital::VITAL_META_GPS_SEC ).as_double();
      // TODO: also use gps_week and convert to UTC to get abosolute time
      // or subtract off first frame time to get time relative to start
      ts.set_time_seconds( gps_sec );
    }
  }

  return ts;
}

// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
video_input_pos
::frame_image()
{
  return nullptr;
}


// ------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_pos
::frame_metadata()
{
  kwiver::vital::metadata_vector vect;
  if ( d->d_metadata )
  {
    vect.push_back( d->d_metadata );
  }

  return vect;
}

kwiver::vital::metadata_map_sptr
video_input_pos
::metadata_map()
{
  if ( !d->d_have_metadata_map )
  {
    kwiver::vital::timestamp::frame_t fn = 0;
    for (const auto& f: d->d_img_md_files)
    {
      ++fn;
      kwiver::vital::timestamp ts;
      auto md = d->process_metadata(f, fn, ts);
      if (md)
      {
        vital::metadata_vector mdv(1, md);
        std::pair<vital::timestamp::frame_t, vital::metadata_vector> el(fn, mdv);
        d->d_metadata_map.insert( el );
      }
    }

    d->d_have_metadata_map = true;
  }

  return std::make_shared<kwiver::vital::simple_metadata_map>(d->d_metadata_map);
}

} } }     // end namespace
