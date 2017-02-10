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

#include "ins_data_reader.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/exceptions.h>
#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>
#include <vital/types/geo_lat_lon.h>

#include <vital/video_metadata/video_metadata.h>
#include <vital/video_metadata/video_metadata_traits.h>

#include <kwiversys/SystemTools.hxx>

#include <vector>
#include <stdint.h>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace core {

class ins_data_reader::priv
{
public:
  priv()
    : c_start_at_frame( 1 )
    , c_stop_after_frame( 0 )
    , c_frame_time( 0.03333 )
    , d_at_eov( false )
  { }

  // Configuration values
  unsigned int c_start_at_frame;
  unsigned int c_stop_after_frame;
  std::string c_meta_directory;
  std::string c_image_list_file;
  float c_frame_time;

  // local state
  bool d_at_eov;

  std::vector < kwiver::vital::path_t > d_metadata_files;
  std::vector < kwiver::vital::path_t >::const_iterator d_current_file;
  std::vector < kwiver::vital::path_t >::const_iterator d_end;
  kwiver::vital::timestamp::frame_t d_frame_number;
  kwiver::vital::timestamp::time_t d_frame_time;

  vital::video_metadata_sptr d_metadata;
};


// ------------------------------------------------------------------
ins_data_reader
::ins_data_reader()
  : d( new ins_data_reader::priv )
{
  attach_logger( "ins_data_reader" );

  set_capability( vital::algo::video_input::HAS_EOV, true );
  set_capability( vital::algo::video_input::HAS_FRAME_NUMBERS, true );
  set_capability( vital::algo::video_input::HAS_FRAME_TIME, true );
  set_capability( vital::algo::video_input::HAS_METADATA, true );

  set_capability( vital::algo::video_input::HAS_FRAME_DATA, false );
  set_capability( vital::algo::video_input::HAS_ABSOLUTE_FRAME_TIME, false ); // MAYBE
  set_capability( vital::algo::video_input::HAS_TIMEOUT, false );
}


// ------------------------------------------------------------------
ins_data_reader
::~ins_data_reader()
{
}


// ------------------------------------------------------------------
ins_data_reader
::ins_data_reader( ins_data_reader const& other )
  : d( new ins_data_reader::priv() )
{
  // copy CTOR

  // TBD
}


// ------------------------------------------------------------------
vital::config_block_sptr
ins_data_reader
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

  return config;
}


// ------------------------------------------------------------------
void
ins_data_reader
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
}


// ------------------------------------------------------------------
bool
ins_data_reader
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// ------------------------------------------------------------------
void
ins_data_reader
::open( std::string image_list_name )
{
  // open file and read lines
  std::ifstream ifs( image_list_name.c_str() );
  if ( ! ifs )
  {
    throw kwiver::vital::invalid_file( image_list_name, "Could not open file" );
  }

  kwiver::vital::data_stream_reader stream_reader( ifs );

  // verify and get file names in a list
  for ( std::string line; stream_reader.getline( line ); /* null */ )
  {
    // Get base name from file
    std::string resolved_file = d->c_meta_directory;
    resolved_file += "/" + kwiversys::SystemTools::GetFilenameWithoutExtension( line ) + ".txt";
    if ( ! kwiversys::SystemTools::FileExists( resolved_file ) )
    {
      LOG_DEBUG( m_logger, "Could not find file " << resolved_file
                 <<". This frame will not have any metadata." );
      resolved_file.clear(); // indicate that the metadata file could not be found
    }

    d->d_metadata_files.push_back( resolved_file );
  } // end for

  d->d_current_file = d->d_metadata_files.begin();
  d->d_frame_number = 1;

  if ( d->c_start_at_frame > 1 )
  {
    d->d_current_file +=  d->c_start_at_frame - 1;
    d->d_frame_number +=  d->c_start_at_frame - 1;
  }

  d->d_end = d->d_metadata_files.end(); // set default end marker

  if ( d->c_stop_after_frame > 0 )
  {
    if ( ( d->d_frame_number + d->c_stop_after_frame ) < d->d_metadata_files.size() )
    {
      d->d_end = d->d_current_file + d->c_stop_after_frame;
    }
  }
}


// ------------------------------------------------------------------
void
ins_data_reader
::close()
{
}


// ------------------------------------------------------------------
bool
ins_data_reader
::end_of_video() const
{
  return d->d_at_eov;
}


// ------------------------------------------------------------------
bool
ins_data_reader
::good() const
{
  // This could use a more nuanced approach
  return true;
}


// ------------------------------------------------------------------
bool
ins_data_reader
::next_frame( kwiver::vital::timestamp& ts,   // returns timestamp
              uint32_t                  timeout ) // not supported
{
  // Check for at end of data
  if ( d->d_at_eov )
  {
    return false;
  }

  if ( d->d_current_file == d->d_end )
  {
    d->d_at_eov = true;
    return false;
  }

  // reset current metadata packet.
  d->d_metadata = 0;

  if ( ! d->d_current_file->empty() )
  {
    // Open next file in the list
    std::ifstream in_stream( *d->d_current_file );
    if ( ! in_stream )
    {
      // should never happen since the file was pre-verified
      throw kwiver::vital::file_not_found_exception( *d->d_current_file, "could not locate file" );
    }

    std::string line;
    getline( in_stream, line );

    // Tokenize the string
    std::vector< std::string > tokens;
    kwiver::vital::tokenize( line, tokens, ",", true );

    unsigned int base = 0;

    // some POS files do not have the source name
    if ( ( tokens.size() < 14 ) || ( tokens.size() > 15 ) )
    {
      std::ostringstream ss;
      ss  << "Too few fields found in file "
          << *d->d_current_file
          << "  (discovered " << tokens.size() << " field(s), expected "
          << "14 or 15).";
      throw vital::invalid_data( ss.str() );
    }

    // make a new metadata container.
    d->d_metadata = std::make_shared<kwiver::vital::video_metadata>();
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_METADATA_ORIGIN, std::string( "POS-file") ) );

    if ( tokens.size() == 15 )
    {
      base = 1;
      d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_IMAGE_SOURCE_SENSOR, tokens[0] ) );
    }
    else
    {
      // Set name to "MAPTK"
      d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_IMAGE_SOURCE_SENSOR, std::string( "MAPTK" ) ) );
    }

    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_SENSOR_YAW_ANGLE, tokens[base + 0] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_SENSOR_PITCH_ANGLE, tokens[ base + 1] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_SENSOR_ROLL_ANGLE, tokens[base + 2] ) );

    kwiver::vital::geo_lat_lon latlon( std::stod( tokens[ base + 3]), std::stod( tokens[ base + 4 ] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_SENSOR_LOCATION, latlon ) );

    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_SENSOR_ALTITUDE, tokens[base + 5] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_GPS_SEC, tokens[base + 6] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_GPS_WEEK, tokens[base + 7] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_NORTHING_VEL, tokens[base + 8] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_EASTING_VEL, tokens[base + 9] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_UP_VEL, tokens[base + 10] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_IMU_STATUS, tokens[base + 11] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_LOCAL_ADJ, tokens[base + 12] ) );
    d->d_metadata->add( NEW_METADATA_ITEM( kwiver::vital::VITAL_META_DST_FLAGS, tokens[base + 13] ) );

    // Return timestamp
    ts = kwiver::vital::timestamp( d->d_frame_time, d->d_frame_number );
    d->d_metadata->set_timestamp( ts );
  }

  // update timestamp
  ++d->d_frame_number;
  ++d->d_current_file;
  d->d_frame_time += d->c_frame_time;

  return true;
} // ins_data_reader::next_frame


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
ins_data_reader
::frame_image()
{
  // Could return a blank image, but we do not know a good size;
  return 0;
}


// ------------------------------------------------------------------
kwiver::vital::video_metadata_vector
ins_data_reader
::frame_metadata()
{
  kwiver::vital::video_metadata_vector vect;
  if ( d->d_metadata )
  {
    vect.push_back( d->d_metadata );
  }

  return vect;
}

} } }     // end namespace
