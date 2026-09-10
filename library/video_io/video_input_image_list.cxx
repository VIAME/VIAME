// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "video_input_image_list.h"

#include <viame/algorithm_framework/algo/image_io.h>

#include <viame/algorithm_framework/util/data_stream_reader.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/util/tokenize.h>

#include <viame/core_types/image.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/metadata_traits.h>
#include <viame/core_types/timestamp.h>

#include <viame/algorithm_framework/exceptions.h>
#include <viame/core_types/vital_types.h>

#include <viame/algorithm_framework/range/iota.h>

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

namespace {

std::string const SEP_PATH{ ":" };
std::string const SEP_EXTS{ ";" };

} // namespace anonymous

// ----------------------------------------------------------------------------
class video_input_image_list::priv
{
public:
  priv( video_input_image_list& parent )
    : m_parent{ parent },
      m_current_file{ m_files.end() }
  {}

  video_input_image_list& m_parent;

  // we need to recalculate these values each time because between invocations
  // of set/get configuration parent.set_path() will make these out of sync.
  std::vector< std::string >
  c_search_path()
  {
    auto const& path = m_parent.get_path();
    std::vector< std::string > result;
    kv::tokenize( path, result, SEP_PATH, kv::TokenizeTrimEmpty );
    result.push_back( "." );
    return result;
  }

  std::vector< std::string >
  c_allowed_extensions()
  {
    auto const& extensions = m_parent.get_allowed_extensions();
    std::vector< std::string > result;
    kv::tokenize( extensions, result, SEP_EXTS, kv::TokenizeTrimEmpty );
    return result;
  }

  bool c_sort_by_time() { return m_parent.get_sort_by_time(); }

  bool c_skip_bad_images() { return m_parent.get_skip_bad_images(); }

  bool
  c_disable_image_load() { return m_parent.get_disable_image_load(); }

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
  vital::algo::image_io_sptr
  m_image_reader()
  {
    return m_parent.get_image_reader();
  }

  void read_from_file( std::string const& filename );
  void read_from_directory( std::string const& dirname );
  void sort_by_time( std::vector< kv::path_t >& files );
  vital::metadata_sptr frame_metadata(
    kv::path_t const& file, kv::image_container_sptr image = nullptr );
};

// ----------------------------------------------------------------------------
void
video_input_image_list
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "arrows.core.video_input_image_list" );

  set_capability( video_input::HAS_EOV, true );
  set_capability( video_input::HAS_FRAME_NUMBERS, true );
  set_capability( video_input::HAS_FRAME_DATA, true );
  set_capability( video_input::HAS_METADATA, true );

  set_capability( video_input::HAS_FRAME_TIME, false );
  set_capability( video_input::HAS_ABSOLUTE_FRAME_TIME, false );
  set_capability( video_input::HAS_TIMEOUT, false );
  set_capability( video_input::IS_SEEKABLE_BY_FRAME, true );
  set_capability( video_input::IS_SEEKABLE_BY_TIME, false );
}

// ----------------------------------------------------------------------------
video_input_image_list
::~video_input_image_list()
{}

// ----------------------------------------------------------------------------
void
video_input_image_list
::set_configuration_internal( vital::config_block_sptr in_config )
{
  auto const& config = this->get_configuration();
  config->merge_config( in_config );

  // Check capabilities of image reader
  if( d->m_image_reader() != nullptr )
  {
    auto const& reader_capabilities =
      d->m_image_reader()->get_implementation_capabilities();
    set_capability(
      HAS_FRAME_TIME,
      reader_capabilities.capability( image_io::HAS_TIME ) );
  }
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::check_configuration( vital::config_block_sptr config ) const
{
  // Check the reader configuration.
  return kwiver::vital::check_nested_algo_configuration< image_io >(
    "image_reader", config );
}

// ----------------------------------------------------------------------------
void
video_input_image_list
::open( std::string list_name )
{
  // Close the video in case already open
  this->close();

  if( !d->m_image_reader() )
  {
    VITAL_THROW(
      kv::algorithm_configuration_exception,
      interface_name(), plugin_name(), "invalid image_reader." );
  }

  if( ksst::FileIsDirectory( list_name ) )
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
size_t
video_input_image_list
::num_frames() const
{
  return d->m_files.size();
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::next_frame( kv::time_usec_t /*timeout*/ )
{
  if( this->end_of_video() )
  {
    return false;
  }

  // Clear the last loaded image
  d->m_image = nullptr;

  // If this is the first call to next_frame(), do not increment
  // the file iteration; next_frame() must be called once
  // before accessing the first frame
  if( d->m_frame_number > 0 )
  {
    ++d->m_current_file;
  }

  ++d->m_frame_number;

  return !this->end_of_video();
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::seek_frame(
  kv::timestamp::frame_t frame_number,
  kv::time_usec_t /*timeout*/ )
{
  // Check if requested frame exists
  auto const max_frame_number =
    static_cast< kv::timestamp::frame_t >( d->m_files.size() );
  if( frame_number > max_frame_number || frame_number <= 0 )
  {
    return false;
  }

  // Adjust frame number if this is the first call to seek_frame or next_frame
  if( d->m_frame_number == 0 )
  {
    d->m_frame_number = 1;
  }

  // Calculate distance to new frame
  auto const frame_diff = frame_number - d->m_frame_number;
  d->m_current_file += frame_diff;
  d->m_frame_number = frame_number;

  // Clear the last loaded image
  d->m_image = nullptr;

  return !this->end_of_video();
}

// ----------------------------------------------------------------------------
bool
video_input_image_list
::seek_time(
  [[maybe_unused]] vital::timestamp::time_t time_usec,
  [[maybe_unused]] vital::time_usec_t timeout )
{
  // TODO: Unimplemented
  return false;
}

// ----------------------------------------------------------------------------
kv::timestamp
video_input_image_list
::frame_timestamp() const
{
  if( this->end_of_video() )
  {
    return {};
  }

  kv::timestamp ts;

  ts.set_frame( d->m_frame_number );

  auto const& reader_capabilities =
    d->m_image_reader()->get_implementation_capabilities();
  if( reader_capabilities.capability( image_io::HAS_TIME ) )
  {
    auto const& md = d->frame_metadata( *d->m_current_file, d->m_image );
    if( md )
    {
      auto const& mdts = md->timestamp();
      if( mdts.has_valid_time() )
      {
        ts.set_time_usec( mdts.get_time_usec() );
      }
    }
  }

  return ts;
}

// ----------------------------------------------------------------------------
kv::path_t
video_input_image_list
::filename() const
{
  // viame/master dereferenced this unguarded; check against the end iterator
  // so a call before open() or after the last frame cannot be UB.
  if( d->m_current_file != d->m_files.cend() )
  {
    return *( d->m_current_file );
  }
  return {};
}

// ----------------------------------------------------------------------------
kv::image_container_sptr
video_input_image_list
::frame_image()
{
  if( !d->m_image && this->good() )
  {
    LOG_DEBUG(
      logger(),
      "reading image from file \"" << *d->m_current_file << "\"" );

    // Read image file
    //
    // This call returns a *new* image container; this is good since
    // we are going to pass it downstream using the sptr
    if( d->c_disable_image_load() )
    {
      d->m_image = kv::image_container_sptr();
    }
    else if( !d->c_skip_bad_images() )
    {
      d->m_image = d->m_image_reader()->load( *d->m_current_file );
    }
    else
    {
      // The first frame still fails hard, as an extra safety check that the
      // list points at readable imagery at all.
      try
      {
        d->m_image = d->m_image_reader()->load( *d->m_current_file );
      }
      catch( ... )
      {
        if( d->m_frame_number <= 1 )
        {
          throw;
        }
        LOG_WARN(
          logger(),
          "skipping unreadable image \"" << *d->m_current_file << "\"" );
        d->m_image = kv::image_container_sptr();
      }
    }
  }
  return d->m_image;
}

// ----------------------------------------------------------------------------
kv::metadata_vector
video_input_image_list
::frame_metadata()
{
  if( !this->good() )
  {
    return {};
  }

  return { 1, d->frame_metadata( *d->m_current_file, d->m_image ) };
}

// ----------------------------------------------------------------------------
void
video_input_image_list::priv
::read_from_file( std::string const& filename )
{
  // Open file and read lines
  std::ifstream ifs( filename.c_str() );
  if( !ifs )
  {
    VITAL_THROW( kv::invalid_file, filename, "Could not open file" );
  }

  std::vector< std::string > search_path = this->c_search_path();

  // Add directory that contains the list file to the path
  auto const& list_path = ksst::GetFilenamePath( filename );
  if( !list_path.empty() )
  {
    search_path.push_back( list_path );
  }

  kv::data_stream_reader stream_reader( ifs );

  // Verify and get file names in a list
  auto data_dir = std::string{};
  auto line = std::string{};

  // Read the first line and determine to file location
  if( stream_reader.getline( line ) )
  {
    auto resolved_file = line;
    if( !ksst::FileExists( resolved_file ) && !this->c_disable_image_load() )
    {
      // Resolve against specified path
      resolved_file = ksst::FindFile( line, search_path, true );
      if( resolved_file.empty() )
      {
        VITAL_THROW(
          kv::file_not_found_exception, line,
          "could not locate file in path" );
      }
      if( ksst::StringEndsWith( resolved_file.c_str(), line.c_str() ) )
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
  while( stream_reader.getline( line ) )
  {
    auto resolved_file = line;
    if( !ksst::FileExists( resolved_file ) && !this->c_disable_image_load() )
    {
      resolved_file = data_dir + line;
      if( !ksst::FileExists( resolved_file ) )
      {
        VITAL_THROW(
          kv::file_not_found_exception, line,
          "could not locate file relative to \"" +
          data_dir + "\"" );
      }
    }

    this->m_files.push_back( resolved_file );
  }

  if( c_sort_by_time() )
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

    bool
    operator<( entry const& other ) const
    {
      return this->time < other.time;
    }
  };

  auto scratch = std::vector< entry >{};
  scratch.reserve( files.size() );

  for( auto& file : files )
  {
    auto const& md = this->m_image_reader()->load_metadata( file );

    if( !md || !md->timestamp().has_valid_time() )
    {
      VITAL_THROW( kv::invalid_file, file, "Could not load time" );
    }

    scratch.push_back(
      { md->timestamp().get_time_usec(),
        std::move( file ) } );
  }

  std::sort( scratch.begin(), scratch.end() );

  files.clear();
  files.reserve( scratch.size() );

  for( auto& file : scratch )
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
  if( !directory.Load( dirname ) )
  {
    VITAL_THROW(
      kv::invalid_file, dirname,
      "Could not open directory" );
  }

  // Read each entry
  for( auto const i : kvr::iota( directory.GetNumberOfFiles() ) )
  {
    auto const filename = std::string{ directory.GetFile( i ) };
    auto const& resolved_file = dirname + "/" + filename;

    if( !ksst::FileExists( resolved_file ) )
    {
      VITAL_THROW(
        kv::file_not_found_exception, filename,
        "could not locate file in path" );
    }
    if( !ksst::FileIsDirectory( resolved_file ) )
    {
      if( this->c_allowed_extensions().empty() )
      {
        this->m_files.push_back( resolved_file );
      }
      else
      {
        for( auto const& extension : this->c_allowed_extensions() )
        {
          std::string resolved_lower = ksst::LowerCase( resolved_file );
          std::string extension_lower = ksst::LowerCase( extension );
          if( ksst::StringEndsWith(
            resolved_lower,
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
  if( c_sort_by_time() )
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
::frame_metadata(
  kv::path_t const& file,
  kv::image_container_sptr image )
{
  auto it = m_metadata_by_path.find( file );
  if( it != m_metadata_by_path.end() )
  {
    return it->second;
  }

  kv::metadata_sptr md;
  if( image )
  {
    md = image->get_metadata();
  }
  if( !md )
  {
    md = m_image_reader()->load_metadata( file );
  }
  if( !md )
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
