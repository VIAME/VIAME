/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Ingest descriptors from a pipeline and write to file
 */

#include "ingest_descriptors_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor_set.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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

create_config_trait( output_file, std::string, "descriptors.csv",
  "Path to the output file for storing descriptors with UIDs" );
create_config_trait( max_frame_buffer, unsigned, "0",
  "Maximum number of frames to buffer descriptors over for larger batch sizes" );
create_config_trait( max_descriptor_buffer, unsigned, "10000",
  "Maximum number of descriptors to buffer before writing to file" );
create_config_trait( append_mode, bool, "false",
  "If true, append to existing file. If false, overwrite." );

//--------------------------------------------------------------------------------
// Private implementation class
class ingest_descriptors_process::priv
{
public:
  priv()
    : m_output_file( "descriptors.csv" )
    , m_max_frame_buffer( 0 )
    , m_max_descriptor_buffer( 10000 )
    , m_append_mode( false )
    , m_frame_counter( 0 )
    , m_file_opened( false ) {}

  ~priv()
  {
    if( m_writer.is_open() )
    {
      m_writer.close();
    }
  }

  std::string m_output_file;
  unsigned m_max_frame_buffer;
  unsigned m_max_descriptor_buffer;
  bool m_append_mode;

  unsigned m_frame_counter;
  bool m_file_opened;

  std::ofstream m_writer;

  // Buffer for descriptors: pairs of (UID, descriptor vector)
  std::vector< std::pair< std::string, std::vector< double > > > m_descriptor_buffer;
};

// ===============================================================================

ingest_descriptors_process
::ingest_descriptors_process( config_block_sptr const& config )
  : process( config ),
    d( new ingest_descriptors_process::priv() )
{
  make_ports();
  make_config();
}


ingest_descriptors_process
::~ingest_descriptors_process()
{
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::_configure()
{
  d->m_output_file = config_value_using_trait( output_file );
  d->m_max_frame_buffer = config_value_using_trait( max_frame_buffer );
  d->m_max_descriptor_buffer = config_value_using_trait( max_descriptor_buffer );
  d->m_append_mode = config_value_using_trait( append_mode );

  // Create output directory if it doesn't exist
  filesystem::path output_path( d->m_output_file );
  filesystem::path parent_path = output_path.parent_path();

  if( !parent_path.empty() && !filesystem::exists( parent_path ) )
  {
    filesystem::create_directories( parent_path );
  }

  // Open output file
  std::ios_base::openmode mode = std::ofstream::out;
  if( d->m_append_mode )
  {
    mode |= std::ofstream::app;
  }

  d->m_writer.open( d->m_output_file, mode );

  if( !d->m_writer.is_open() )
  {
    throw std::runtime_error( "Failed to open output file: " + d->m_output_file );
  }

  d->m_file_opened = true;
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::_step()
{
  // Grab input values from ports
  kwiver::vital::descriptor_set_sptr vital_descriptor_set =
    grab_from_port_using_trait( descriptor_set );

  kwiver::vital::string_vector_sptr string_tuple =
    grab_from_port_using_trait( string_vector );

  // Validate input sizes match
  if( vital_descriptor_set->size() != string_tuple->size() )
  {
    std::ostringstream ss;
    ss << "Received an incongruent pair of descriptors and UID labels ("
       << vital_descriptor_set->size() << " descriptors vs. "
       << string_tuple->size() << " uids)";
    throw std::runtime_error( ss.str() );
  }

  // Convert descriptors to buffer entries
  std::vector< kwiver::vital::descriptor_sptr > descriptors =
    vital_descriptor_set->descriptors();

  for( size_t i = 0; i < descriptors.size(); ++i )
  {
    const std::string& uid_str = (*string_tuple)[i];
    std::vector< double > desc_vector = descriptors[i]->as_double();

    d->m_descriptor_buffer.emplace_back( uid_str, std::move( desc_vector ) );
  }

  // Determine if we need to write out a new batch
  if( d->m_descriptor_buffer.size() >= d->m_max_descriptor_buffer ||
      ( d->m_max_frame_buffer > 0 && d->m_frame_counter >= d->m_max_frame_buffer ) )
  {
    flush_buffer();
  }

  d->m_frame_counter++;

  // Pass on input descriptors and UIDs
  push_to_port_using_trait( descriptor_set, vital_descriptor_set );
  push_to_port_using_trait( string_vector, string_tuple );
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::_finalize()
{
  // Flush any remaining buffered descriptors
  if( !d->m_descriptor_buffer.empty() )
  {
    flush_buffer();
  }

  if( d->m_writer.is_open() )
  {
    d->m_writer.close();
  }
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::flush_buffer()
{
  if( !d->m_writer.is_open() )
  {
    return;
  }

  // Write each buffered descriptor as a CSV line: uid,val1,val2,...,valN
  for( const auto& entry : d->m_descriptor_buffer )
  {
    const std::string& uid = entry.first;
    const std::vector< double >& desc = entry.second;

    d->m_writer << uid;

    for( double val : desc )
    {
      d->m_writer << "," << val;
    }

    d->m_writer << "\n";
  }

  d->m_writer.flush();

  d->m_frame_counter = 0;
  d->m_descriptor_buffer.clear();
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::make_ports()
{
  sprokit::process::port_flags_t optional;

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( descriptor_set, required );
  declare_input_port_using_trait( string_vector, required );

  // -- outputs --
  declare_output_port_using_trait( descriptor_set, optional );
  declare_output_port_using_trait( string_vector, optional );
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_process
::make_config()
{
  declare_config_using_trait( output_file );
  declare_config_using_trait( max_frame_buffer );
  declare_config_using_trait( max_descriptor_buffer );
  declare_config_using_trait( append_mode );
}

} // end namespace core

} // end namespace viame
