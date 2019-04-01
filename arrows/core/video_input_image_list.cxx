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

#include <kwiversys/Directory.hxx>
#include <kwiversys/SystemTools.hxx>

#include <algorithm>
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
  priv( video_input_image_list* parent )
  : m_parent( parent )
  , c_sort_by_time( false )
  , m_current_file( m_files.end() )
  , m_frame_number( 0 )
  , m_image( nullptr )
  , m_have_metadata_map( false )
  {}

  video_input_image_list* m_parent;

  // Configuration values
  std::vector< std::string > c_search_path;
  std::vector< std::string > c_allowed_extensions;
  bool c_sort_by_time;

  // local state
  std::vector < kwiver::vital::path_t > m_files;
  std::vector < kwiver::vital::path_t >::const_iterator m_current_file;
  kwiver::vital::frame_id_t m_frame_number;
  kwiver::vital::image_container_sptr m_image;

  // metadata map
  bool m_have_metadata_map;
  vital::metadata_map::map_metadata_t m_metadata_map;
  std::map< kwiver::vital::path_t, kwiver::vital::metadata_sptr > m_metadata_by_path;

  // processing classes
  vital::algo::image_io_sptr m_image_reader;

  void read_from_file( std::string const& filename );
  void read_from_directory( std::string const& dirname );
  void sort_by_time( std::vector < kwiver::vital::path_t >& files );
  vital::metadata_sptr frame_metadata(
    const kwiver::vital::path_t& file,
    kwiver::vital::image_container_sptr image = nullptr );
};


// ----------------------------------------------------------------------------
video_input_image_list
::video_input_image_list()
  : d( new video_input_image_list::priv( this ) )
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


// ----------------------------------------------------------------------------
video_input_image_list
::~video_input_image_list()
{
}


// ----------------------------------------------------------------------------
vital::config_block_sptr
video_input_image_list
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::video_input::get_configuration();

  config->set_value( "path", "",
                     "Path to search for image file. "
                     "If a file name is not absolute, this list of directories "
                     "is scanned to find the file. The current directory '.' is "
                     "automatically appended to the end of the path. The format "
                     "of this path is the same as the standard path specification, "
                     "a set of directories separated by a colon (':')" );
  config->set_value( "allowed_extensions", "",
                     "Semicolon-separated list of allowed file extensions. "
                     "Leave empty to allow all file extensions." );
  config->set_value( "sort_by_time", "false",
                     "Instead of accepting the input list as-is, sort input file "
                     "order based on timestamp metadata contained either in the "
                     "file or in each filename." );

  vital::algo::image_io::
    get_nested_algo_configuration( "image_reader", config, d->m_image_reader );

  return config;
}


// ----------------------------------------------------------------------------
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

  // Create vector of allowed file extensions
  std::string extensions = config->get_value<std::string>( "allowed_extensions", "" );
  kwiver::vital::tokenize( extensions, d->c_allowed_extensions, ";", kwiver::vital::TokenizeTrimEmpty );

  // Read standalone variables
  d->c_sort_by_time = config->get_value<bool>( "sort_by_time" );

  // Setup actual reader algorithm
  vital::algo::image_io::
    set_nested_algo_configuration( "image_reader", config, d->m_image_reader );

  // Check capabilities of image reader
  set_capability( vital::algo::video_input::HAS_FRAME_TIME,
    d->m_image_reader->get_implementation_capabilities().capability(
      vital::algo::image_io::HAS_TIME ) );
}


// ----------------------------------------------------------------------------
bool
video_input_image_list
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the reader configuration.
  return vital::algo::image_io::
    check_nested_algo_configuration( "image_reader", config );
}


// ----------------------------------------------------------------------------
void
video_input_image_list
::open( std::string list_name )
{
  typedef kwiversys::SystemTools ST;

  // close the video in case already open
  this->close();

  if ( ! d->m_image_reader )
  {
    VITAL_THROW( kwiver::vital::algorithm_configuration_exception, type_name(), impl_name(),
          "invalid image_reader." );
  }

  if( ST::FileIsDirectory( list_name ) )
  {
    d->read_from_directory( list_name );
  }
  else
  {
    d->read_from_file( list_name );
  }

  d->m_current_file = d->m_files.begin();
  d->m_frame_number = 0;
}


// ----------------------------------------------------------------------------
void
video_input_image_list
::close()
{
  d->m_files.clear();
  d->m_current_file = d->m_files.end();
  d->m_frame_number = 0;
  d->m_image = nullptr;
}


// ----------------------------------------------------------------------------
bool
video_input_image_list
::end_of_video() const
{
  return ( d->m_current_file == d->m_files.end() );
}


// ----------------------------------------------------------------------------
bool
video_input_image_list
::good() const
{
  return d->m_frame_number > 0 && ! this->end_of_video();
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::seekable() const
{
  return true;
}

// ----------------------------------------------------------------------------
size_t
video_input_image_list
::num_frames() const
{
  return d->m_files.size();
}

// ----------------------------------------------------------------------------
kwiver::vital::path_t
video_input_image_list
::filename() const
{
  return *(d->m_current_file);
}

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
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
  ts = this->frame_timestamp();

  return ! this->end_of_video();
}

// ----------------------------------------------------------------------------
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

  if ( d->m_image_reader->get_implementation_capabilities().capability(
      kwiver::vital::algo::image_io::HAS_TIME ) )
  {
    auto md = d->frame_metadata( *d->m_current_file, d->m_image );
    if ( md )
    {
      auto mdts = md->timestamp();
      if ( mdts.has_valid_time() )
      {
        ts.set_time_usec( mdts.get_time_usec() );
      }
    }
  }

  return ts;
}


// ----------------------------------------------------------------------------
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


// ----------------------------------------------------------------------------
kwiver::vital::metadata_vector
video_input_image_list
::frame_metadata()
{
  if ( ! this->good() )
  {
    return vital::metadata_vector();
  }
  auto md = d->frame_metadata( *d->m_current_file, d->m_image );
  vital::metadata_vector mdv(1, md);
  return mdv;
}


// ----------------------------------------------------------------------------
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
      auto md = d->frame_metadata( f );
      vital::metadata_vector mdv(1, md);
      d->m_metadata_map[fn] = mdv;
    }

    d->m_have_metadata_map = true;
  }

  return std::make_shared<kwiver::vital::simple_metadata_map>(d->m_metadata_map);
}


// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::read_from_file( std::string const& filename )
{
  typedef kwiversys::SystemTools ST;

  // open file and read lines
  std::ifstream ifs( filename.c_str() );
  if ( ! ifs )
  {
    VITAL_THROW( kwiver::vital::invalid_file, filename, "Could not open file" );
  }

  // Add directory that contains the list file to the path
  std::string list_path = ST::GetFilenamePath( filename );
  if ( ! list_path.empty() )
  {
    this->c_search_path.push_back( list_path );
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
      resolved_file = ST::FindFile( line, this->c_search_path, true );
      if ( resolved_file.empty() )
      {
        VITAL_THROW( kwiver::vital::
          file_not_found_exception, line, "could not locate file in path" );
      }
      if( ST::StringEndsWith( resolved_file.c_str(), line.c_str() ) )
      {
        // extract the prefix added to get the full path
        data_dir = resolved_file.substr(0, resolved_file.size() - line.size() );
      }
    }
    this->m_files.push_back( resolved_file );
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
        VITAL_THROW( kwiver::vital::
          file_not_found_exception, line,
              "could not locate file relative to \"" + data_dir + "\"" );
      }
    }

    this->m_files.push_back( resolved_file );
  } // end while

  if ( c_sort_by_time )
  {
    sort_by_time( this->m_files );
  }
}


// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::sort_by_time( std::vector< kwiver::vital::path_t >& files )
{
  std::map< kwiver::vital::time_usec_t, kwiver::vital::path_t > m_sorted_files;

  for ( auto file : this->m_files )
  {
    kwiver::vital::metadata_sptr md = m_image_reader->load_metadata( file );

    if ( !md || !md->timestamp().has_valid_time() )
    {
      VITAL_THROW( kwiver::vital::invalid_file, file, "Could not load time" );
    }

    m_sorted_files[ md->timestamp().get_time_usec() ] = file;
  }

  files.clear();

  for ( auto file : m_sorted_files )
  {
    files.push_back( file.second );
  }
}


// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::read_from_directory( std::string const& dirname )
{
  typedef kwiversys::SystemTools ST;

  // Open the directory and read the entries
  kwiversys::Directory directory;
  if( ! directory.Load( dirname ) )
  {
    VITAL_THROW( kwiver::vital::invalid_file, dirname, "Could not open directory" );
  }

  // Read each entry
  for( unsigned long i = 0; i < directory.GetNumberOfFiles(); i++ )
  {
    std::string filename = directory.GetFile( i );
    std::string resolved_file = dirname + "/" + filename;
    if( ! ST::FileExists( resolved_file ) )
    {
      VITAL_THROW( kwiver::vital::
        file_not_found_exception, filename, "could not locate file in path" );
    }
    if( ! ST::FileIsDirectory( resolved_file ) )
    {
      if( this->c_allowed_extensions.empty() )
      {
        this->m_files.push_back( resolved_file );
      }
      else
      {
        for( auto const& extension: this->c_allowed_extensions )
        {
          std::string resolved_lower = ST::LowerCase( resolved_file );
          std::string extension_lower = ST::LowerCase( extension );
          if( ST::StringEndsWith( resolved_lower, extension_lower.c_str() ) )
          {
            this->m_files.push_back( resolved_file );
            break;
          }
        }
      }
    }
  }

  // Sort the list
  if ( c_sort_by_time )
  {
    sort_by_time( this->m_files );
  }
  else
  {
    std::sort( this->m_files.begin(), this->m_files.end() );
  }
}


// ----------------------------------------------------------------------------
kwiver::vital::metadata_sptr
video_input_image_list::priv
::frame_metadata( const kwiver::vital::path_t& file,
                  kwiver::vital::image_container_sptr image )
{
  auto it = m_metadata_by_path.find( file );
  if ( it != m_metadata_by_path.end() )
  {
    return it->second;
  }

  kwiver::vital::metadata_sptr md;
  if ( image )
  {
    md = image->get_metadata();
  }
  if ( ! md )
  {
    md = m_image_reader->load_metadata( file );
  }
  if ( ! md )
  {
    md = std::make_shared<kwiver::vital::metadata>();
  }

  md->add( NEW_METADATA_ITEM( vital::VITAL_META_IMAGE_URI, file ) );

  m_metadata_by_path[file] = md;
  return md;
}

} } }     // end namespace
