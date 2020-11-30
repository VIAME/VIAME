// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "video_input_image_list.h"

#include <vital/algo/image_io.h>

#include <vital/util/data_stream_reader.h>
#include <vital/util/tokenize.h>

#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/timestamp.h>

#include <vital/exceptions.h>
#include <vital/vital_types.h>

#include <vital/range/iota.h>

#include <kwiversys/Directory.hxx>
#include <kwiversys/SystemTools.hxx>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include <cstdint>

namespace kv = kwiver::vital;
namespace kvr = kwiver::vital::range;

using ksst = kwiversys::SystemTools;

using kv::algo::video_input;
using kv::algo::image_io;

namespace kwiver {

namespace arrows {

namespace core {

// ----------------------------------------------------------------------------
class video_input_image_list::priv
{
public:
  priv( video_input_image_list* parent )
    : m_parent{ parent },
      m_current_file{ m_files.end() }
  {}

  video_input_image_list* const m_parent;

  // Configuration values
  std::vector< std::string > c_search_path;
  std::vector< std::string > c_allowed_extensions;
  bool c_sort_by_time = false;

  // Local state
  std::vector< kv::path_t > m_files;
  std::vector< kv::path_t >::const_iterator m_current_file;
  kv::frame_id_t m_frame_number = 0;
  kv::image_container_sptr m_image;

  // Metadata map
  bool m_have_metadata_map = false;
  vital::metadata_map::map_metadata_t m_metadata_map;
  std::map< kv::path_t, kv::metadata_sptr > m_metadata_by_path;

  // Processing classes
  vital::algo::image_io_sptr m_image_reader;

  void read_from_file( std::string const& filename );
  void read_from_directory( std::string const& dirname );
  void sort_by_time( std::vector< kv::path_t >& files );
  vital::metadata_sptr frame_metadata(
    kv::path_t const& file, kv::image_container_sptr image = nullptr );
};

// ----------------------------------------------------------------------------
video_input_image_list
::video_input_image_list()
  : d( new video_input_image_list::priv( this ) )
{
  attach_logger( "arrows.core.video_input_image_list" );

  set_capability( video_input::HAS_EOV, true );
  set_capability( video_input::HAS_FRAME_NUMBERS, true );
  set_capability( video_input::HAS_FRAME_DATA, true );
  set_capability( video_input::HAS_METADATA, true );

  set_capability( video_input::HAS_FRAME_TIME, false );
  set_capability( video_input::HAS_ABSOLUTE_FRAME_TIME, false );
  set_capability( video_input::HAS_TIMEOUT, false );
  set_capability( video_input::IS_SEEKABLE, true );
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
  // Get base configuration from base class
  auto const& config = video_input::get_configuration();

  config->set_value(
    "path", "",
    "Path to search for image file. "
    "If a file name is not absolute, this list of directories is scanned "
    "to find the file. The current directory '.' is automatically appended "
    "to the end of the path. "
    "The format of this path is the same as the standard path specification, "
    "a set of directories separated by a colon (':')" );
  config->set_value(
    "allowed_extensions", "",
    "Semicolon-separated list of allowed file extensions. "
    "Leave empty to allow all file extensions." );
  config->set_value(
    "sort_by_time", "false",
    "Instead of accepting the input list as-is, sort the input file list "
    "based on the timestamp metadata provided for the file." );

  image_io::get_nested_algo_configuration(
    "image_reader", config, d->m_image_reader );

  return config;
}

// ----------------------------------------------------------------------------
void
video_input_image_list
::set_configuration( vital::config_block_sptr in_config )
{
  auto const& config = this->get_configuration();
  config->merge_config( in_config );

  // Extract string and create vector of directories
  auto const& path = config->get_value< std::string >( "path", {} );
  kv::tokenize( path, d->c_search_path, ":", kv::TokenizeTrimEmpty );
  d->c_search_path.push_back( "." ); // Add current directory

  // Create vector of allowed file extensions
  auto const& extensions =
    config->get_value< std::string >( "allowed_extensions", {} );
  kv::tokenize( extensions, d->c_allowed_extensions, ";",
                kv::TokenizeTrimEmpty );

  // Read standalone variables
  d->c_sort_by_time = config->get_value< bool >( "sort_by_time" );

  // Setup actual reader algorithm
  image_io::set_nested_algo_configuration(
    "image_reader", config, d->m_image_reader );

  // Check capabilities of image reader
  auto const& reader_capabilities =
    d->m_image_reader->get_implementation_capabilities();
  set_capability( HAS_FRAME_TIME,
                  reader_capabilities.capability( image_io::HAS_TIME ) );
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the reader configuration.
  return image_io::check_nested_algo_configuration( "image_reader", config );
}

// ----------------------------------------------------------------------------
void
video_input_image_list
::open( std::string list_name )
{
  // Close the video in case already open
  this->close();

  if ( !d->m_image_reader )
  {
    VITAL_THROW( kv::algorithm_configuration_exception,
                 type_name(), impl_name(), "invalid image_reader." );
  }

  if ( ksst::FileIsDirectory( list_name ) )
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
  return d->m_frame_number > 0 && !this->end_of_video();
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
bool
video_input_image_list
::next_frame( kv::timestamp& ts, uint32_t /*timeout*/ )
{
  if ( this->end_of_video() )
  {
    return false;
  }

  // Clear the last loaded image
  d->m_image = nullptr;

  // If this is the first call to next_frame(), do not increment
  // the file iteration; next_frame() must be called once
  // before accessing the first frame
  if ( d->m_frame_number > 0 )
  {
    ++d->m_current_file;
  }

  ++d->m_frame_number;

  // Return timestamp
  ts = this->frame_timestamp();

  return !this->end_of_video();
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::seek_frame( kv::timestamp& ts,
              kv::timestamp::frame_t frame_number,
              uint32_t /*timeout*/ )
{
  // Check if requested frame exists
  auto const max_frame_number =
    static_cast< kv::timestamp::frame_t >( d->m_files.size() );
  if ( frame_number > max_frame_number || frame_number <= 0 )
  {
    return false;
  }

  // Adjust frame number if this is the first call to seek_frame or next_frame
  if ( d->m_frame_number == 0 )
  {
    d->m_frame_number = 1;
  }

  // Calculate distance to new frame
  auto const frame_diff = frame_number - d->m_frame_number;
  d->m_current_file += frame_diff;
  d->m_frame_number = frame_number;

  // Clear the last loaded image
  d->m_image = nullptr;

  // Return timestamp
  ts = this->frame_timestamp();

  return !this->end_of_video();
}

// ----------------------------------------------------------------------------
kv::timestamp
video_input_image_list
::frame_timestamp() const
{
  if ( this->end_of_video() )
  {
    return {};
  }

  kv::timestamp ts;

  ts.set_frame( d->m_frame_number );

  auto const& reader_capabilities =
    d->m_image_reader->get_implementation_capabilities();
  if ( reader_capabilities.capability( image_io::HAS_TIME ) )
  {
    auto const& md = d->frame_metadata( *d->m_current_file, d->m_image );
    if ( md )
    {
      auto const& mdts = md->timestamp();
      if ( mdts.has_valid_time() )
      {
        ts.set_time_usec( mdts.get_time_usec() );
      }
    }
  }

  return ts;
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
video_input_image_list
::frame_image()
{
  if ( !d->m_image && this->good() )
  {
    LOG_DEBUG( logger(),
               "reading image from file \"" << *d->m_current_file << "\"" );

    // Read image file
    //
    // This call returns a *new* image container; this is good since
    // we are going to pass it downstream using the sptr
    d->m_image = d->m_image_reader->load( *d->m_current_file );
  }
  return d->m_image;
}

// ----------------------------------------------------------------------------
kv::metadata_vector
video_input_image_list
::frame_metadata()
{
  if ( !this->good() )
  {
    return {};
  }

  return { 1, d->frame_metadata( *d->m_current_file, d->m_image ) };
}

// ----------------------------------------------------------------------------
kv::metadata_map_sptr
video_input_image_list
::metadata_map()
{
  if ( !d->m_have_metadata_map )
  {
    auto fn = kv::timestamp::frame_t{ 0 };
    for ( auto const& f : d->m_files )
    {
      auto mdv = vital::metadata_vector{ 1, d->frame_metadata( f ) };
      d->m_metadata_map.emplace( ++fn, std::move( mdv ) );
    }

    d->m_have_metadata_map = true;
  }

  return std::make_shared< kv::simple_metadata_map >( d->m_metadata_map );
}

// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::read_from_file( std::string const& filename )
{
  // Open file and read lines
  std::ifstream ifs( filename.c_str() );
  if ( !ifs )
  {
    VITAL_THROW( kv::invalid_file, filename, "Could not open file" );
  }

  // Add directory that contains the list file to the path
  auto const& list_path = ksst::GetFilenamePath( filename );
  if ( !list_path.empty() )
  {
    this->c_search_path.push_back( list_path );
  }

  kv::data_stream_reader stream_reader( ifs );

  // Verify and get file names in a list
  auto data_dir = std::string{};
  auto line = std::string{};

  // Read the first line and determine to file location
  if ( stream_reader.getline( line ) )
  {
    auto resolved_file = line;
    if ( !ksst::FileExists( resolved_file ) )
    {
      // Resolve against specified path
      resolved_file = ksst::FindFile( line, this->c_search_path, true );
      if ( resolved_file.empty() )
      {
        VITAL_THROW( kv:: file_not_found_exception, line,
                     "could not locate file in path" );
      }
      if ( ksst::StringEndsWith( resolved_file.c_str(), line.c_str() ) )
      {
        // extract the prefix added to get the full path
        data_dir =
          resolved_file.substr( 0, resolved_file.size() - line.size() );
      }
    }
    this->m_files.push_back( resolved_file );
  }

  // Read the rest of the file and validate paths
  // Only check the same data_dir used to resolve the first frame
  while ( stream_reader.getline( line ) )
  {
    auto resolved_file = line;
    if ( !ksst::FileExists( resolved_file ) )
    {
      resolved_file = data_dir + line;
      if ( !ksst::FileExists( resolved_file ) )
      {
        VITAL_THROW( kv::file_not_found_exception, line,
                     "could not locate file relative to \"" +
                     data_dir + "\"" );
      }
    }

    this->m_files.push_back( resolved_file );
  }

  if ( c_sort_by_time )
  {
    sort_by_time( this->m_files );
  }
}

// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::sort_by_time( std::vector< kv::path_t >& files )
{
  struct entry
  {
    kv::time_usec_t time;
    kv::path_t path;

    bool operator<( entry const& other ) const
    {
      return this->time < other.time;
    }
  };

  auto scratch = std::vector<entry>{};
  scratch.reserve( files.size() );

  for ( auto& file : files )
  {
    auto const& md = this->m_image_reader->load_metadata( file );

    if ( !md || !md->timestamp().has_valid_time() )
    {
      VITAL_THROW( kv::invalid_file, file, "Could not load time" );
    }

    scratch.push_back( { md->timestamp().get_time_usec(),
                         std::move( file ) } );
  }

  std::sort( scratch.begin(), scratch.end() );

  files.clear();
  files.reserve( scratch.size() );

  for ( auto& file : scratch )
  {
    files.push_back( std::move( file.path ) );
  }
}

// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::read_from_directory( std::string const& dirname )
{
  // Open the directory and read the entries
  kwiversys::Directory directory;
  if ( !directory.Load( dirname ) )
  {
    VITAL_THROW( kv::invalid_file, dirname,
                 "Could not open directory" );
  }

  // Read each entry
  for ( auto const i : kvr::iota( directory.GetNumberOfFiles() ) )
  {
    auto const filename = std::string{ directory.GetFile( i ) };
    auto const& resolved_file = dirname + "/" + filename;

    if ( !ksst::FileExists( resolved_file ) )
    {
      VITAL_THROW( kv::file_not_found_exception, filename,
                   "could not locate file in path" );
    }
    if ( !ksst::FileIsDirectory( resolved_file ) )
    {
      if ( this->c_allowed_extensions.empty() )
      {
        this->m_files.push_back( resolved_file );
      }
      else
      {
        for ( auto const& extension : this->c_allowed_extensions )
        {
          std::string resolved_lower = ksst::LowerCase( resolved_file );
          std::string extension_lower = ksst::LowerCase( extension );
          if ( ksst::StringEndsWith( resolved_lower,
                                     extension_lower.c_str() ) )
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
kv::metadata_sptr
video_input_image_list::priv
::frame_metadata( kv::path_t const& file,
                  kv::image_container_sptr image )
{
  auto it = m_metadata_by_path.find( file );
  if ( it != m_metadata_by_path.end() )
  {
    return it->second;
  }

  kv::metadata_sptr md;
  if ( image )
  {
    md = image->get_metadata();
  }
  if ( !md )
  {
    md = m_image_reader->load_metadata( file );
  }
  if ( !md )
  {
    md = std::make_shared< kv::metadata >();
  }

  md->add< vital::VITAL_META_IMAGE_URI >( file );

  m_metadata_by_path[ file ] = md;
  return md;
}

} // namespace core

} // namespace arrows

} // namespace kwiver
