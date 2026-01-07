/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of CSV descriptor storage backend
 */

#include "store_descriptors_csv.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

#if WIN32 || ( __cplusplus >= 201703L && __has_include(<filesystem>) )
  #include <filesystem>
  namespace filesystem = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem>
  namespace filesystem = std::experimental::filesystem;
#endif

namespace viame
{

namespace core
{

// =============================================================================
// CSV Backend Implementation
// =============================================================================

class csv_descriptor_backend::impl
{
public:
  impl( const std::string& file_path )
    : m_file_path( file_path )
    , m_index_loaded( false )
  {}

  std::string m_file_path;
  std::string m_uid_mapping_file;
  std::string m_track_frame_file;
  std::ofstream m_writer;
  bool m_index_loaded;

  // UID -> descriptor values
  std::unordered_map< std::string, std::vector< double > > m_uid_index;

  // (track_id, frame_id) -> descriptor values (direct mapping)
  using track_frame_key = std::pair< kwiver::vital::track_id_t, kwiver::vital::frame_id_t >;
  std::map< track_frame_key, std::vector< double > > m_track_frame_index;

  // (track_id, frame_id) -> UID mapping
  std::map< track_frame_key, std::string > m_track_frame_to_uid;
};

// -----------------------------------------------------------------------------
csv_descriptor_backend
::csv_descriptor_backend( const std::string& file_path )
  : p( new impl( file_path ) )
{
}

csv_descriptor_backend::~csv_descriptor_backend()
{
  close();
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::open_for_write( bool append )
{
  // Create output directory if it doesn't exist
  filesystem::path output_path( p->m_file_path );
  filesystem::path parent_path = output_path.parent_path();

  if( !parent_path.empty() && !filesystem::exists( parent_path ) )
  {
    filesystem::create_directories( parent_path );
  }

  std::ios_base::openmode mode = std::ofstream::out;
  if( append )
  {
    mode |= std::ofstream::app;
  }

  p->m_writer.open( p->m_file_path, mode );

  if( !p->m_writer.is_open() )
  {
    throw std::runtime_error( "Failed to open output file: " + p->m_file_path );
  }
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::write_descriptor( const std::string& uid, const std::vector< double >& values )
{
  if( !p->m_writer.is_open() )
  {
    return;
  }

  p->m_writer << uid;
  for( double val : values )
  {
    p->m_writer << "," << val;
  }
  p->m_writer << "\n";
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::flush()
{
  if( p->m_writer.is_open() )
  {
    p->m_writer.flush();
  }
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::close()
{
  if( p->m_writer.is_open() )
  {
    p->m_writer.close();
  }
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::load_index()
{
  if( p->m_index_loaded )
  {
    return;
  }

  std::ifstream file( p->m_file_path );

  if( !file.is_open() )
  {
    throw std::runtime_error( "Failed to open descriptor file: " + p->m_file_path );
  }

  std::string line;
  while( std::getline( file, line ) )
  {
    if( line.empty() )
    {
      continue;
    }

    std::istringstream ss( line );
    std::string uid;

    if( !std::getline( ss, uid, ',' ) )
    {
      continue;
    }

    std::vector< double > values;
    std::string value_str;

    while( std::getline( ss, value_str, ',' ) )
    {
      try
      {
        values.push_back( std::stod( value_str ) );
      }
      catch( const std::exception& )
      {
        // Skip invalid values
      }
    }

    if( !values.empty() )
    {
      p->m_uid_index[uid] = std::move( values );
    }
  }

  // Load track_frame file if provided (format: track_id,frame_id,val1,val2,...)
  if( !p->m_track_frame_file.empty() )
  {
    std::ifstream tf_file( p->m_track_frame_file );
    if( tf_file.is_open() )
    {
      while( std::getline( tf_file, line ) )
      {
        if( line.empty() ) continue;

        std::istringstream ss( line );
        std::string track_id_str, frame_id_str;

        if( !std::getline( ss, track_id_str, ',' ) ) continue;
        if( !std::getline( ss, frame_id_str, ',' ) ) continue;

        std::vector< double > values;
        std::string value_str;
        while( std::getline( ss, value_str, ',' ) )
        {
          try { values.push_back( std::stod( value_str ) ); }
          catch( const std::exception& ) {}
        }

        if( !values.empty() )
        {
          try
          {
            kwiver::vital::track_id_t track_id = std::stoll( track_id_str );
            kwiver::vital::frame_id_t frame_id = std::stoll( frame_id_str );
            p->m_track_frame_index[{ track_id, frame_id }] = std::move( values );
          }
          catch( const std::exception& ) {}
        }
      }
    }
  }

  // Load UID mapping file if provided (format: track_id,frame_id,uid)
  if( !p->m_uid_mapping_file.empty() )
  {
    std::ifstream map_file( p->m_uid_mapping_file );
    if( map_file.is_open() )
    {
      while( std::getline( map_file, line ) )
      {
        if( line.empty() ) continue;

        std::istringstream ss( line );
        std::string track_id_str, frame_id_str, uid;

        if( !std::getline( ss, track_id_str, ',' ) ) continue;
        if( !std::getline( ss, frame_id_str, ',' ) ) continue;
        if( !std::getline( ss, uid, ',' ) ) continue;

        try
        {
          kwiver::vital::track_id_t track_id = std::stoll( track_id_str );
          kwiver::vital::frame_id_t frame_id = std::stoll( frame_id_str );
          p->m_track_frame_to_uid[{ track_id, frame_id }] = uid;
        }
        catch( const std::exception& ) {}
      }
    }
  }

  p->m_index_loaded = true;
}

// -----------------------------------------------------------------------------
bool
csv_descriptor_backend
::get_descriptor( const std::string& uid, std::vector< double >& values )
{
  auto it = p->m_uid_index.find( uid );
  if( it != p->m_uid_index.end() )
  {
    values = it->second;
    return true;
  }
  return false;
}

// -----------------------------------------------------------------------------
bool
csv_descriptor_backend
::get_descriptor_by_track_frame(
  kwiver::vital::track_id_t track_id,
  kwiver::vital::frame_id_t frame_id,
  std::vector< double >& values )
{
  impl::track_frame_key key{ track_id, frame_id };

  // First try direct track_frame index
  auto it = p->m_track_frame_index.find( key );
  if( it != p->m_track_frame_index.end() )
  {
    values = it->second;
    return true;
  }

  // Then try UID mapping
  auto uid_it = p->m_track_frame_to_uid.find( key );
  if( uid_it != p->m_track_frame_to_uid.end() )
  {
    return get_descriptor( uid_it->second, values );
  }

  return false;
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::set_uid_mapping_file( const std::string& path )
{
  p->m_uid_mapping_file = path;
}

// -----------------------------------------------------------------------------
void
csv_descriptor_backend
::set_track_frame_file( const std::string& path )
{
  p->m_track_frame_file = path;
}

} // end namespace core
} // end namespace viame
