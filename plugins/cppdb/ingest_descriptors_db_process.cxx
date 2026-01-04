/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Ingest descriptors from a pipeline and write to database
 */

#include "ingest_descriptors_db_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor_set.h>

#include <cppdb/frontend.h>

#include <sstream>
#include <vector>

namespace viame
{

namespace cppdb
{

create_config_trait( conn_str, std::string, "",
  "Database connection string (e.g., postgresql:host=localhost;dbname=viame;user=postgres)" );
create_config_trait( video_name, std::string, "",
  "Video name for associating descriptors with a source" );
create_config_trait( max_frame_buffer, unsigned, "0",
  "Maximum number of frames to buffer descriptors over for larger batch sizes" );
create_config_trait( max_descriptor_buffer, unsigned, "10000",
  "Maximum number of descriptors to buffer before writing to database" );
create_config_trait( commit_interval, unsigned, "1",
  "Number of frames between database commits (0 = commit at end only)" );

//--------------------------------------------------------------------------------
// Private implementation class
class ingest_descriptors_db_process::priv
{
public:
  priv()
    : m_conn_str( "" )
    , m_video_name( "" )
    , m_max_frame_buffer( 0 )
    , m_max_descriptor_buffer( 10000 )
    , m_commit_interval( 1 )
    , m_frame_counter( 0 )
    , m_commit_counter( 0 )
    , m_in_transaction( false ) {}

  ~priv()
  {
    if( m_in_transaction && m_transaction )
    {
      m_transaction->commit();
    }
    if( m_conn.is_open() )
    {
      m_conn.close();
    }
  }

  std::string m_conn_str;
  std::string m_video_name;
  unsigned m_max_frame_buffer;
  unsigned m_max_descriptor_buffer;
  unsigned m_commit_interval;

  unsigned m_frame_counter;
  unsigned m_commit_counter;
  bool m_in_transaction;

  ::cppdb::session m_conn;
  std::unique_ptr< ::cppdb::transaction > m_transaction;

  // Buffer for descriptors: pairs of (UID, descriptor vector)
  std::vector< std::pair< std::string, std::vector< double > > > m_descriptor_buffer;
};

// ===============================================================================

ingest_descriptors_db_process
::ingest_descriptors_db_process( config_block_sptr const& config )
  : process( config ),
    d( new ingest_descriptors_db_process::priv() )
{
  make_ports();
  make_config();
}


ingest_descriptors_db_process
::~ingest_descriptors_db_process()
{
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_db_process
::_configure()
{
  d->m_conn_str = config_value_using_trait( conn_str );
  d->m_video_name = config_value_using_trait( video_name );
  d->m_max_frame_buffer = config_value_using_trait( max_frame_buffer );
  d->m_max_descriptor_buffer = config_value_using_trait( max_descriptor_buffer );
  d->m_commit_interval = config_value_using_trait( commit_interval );

  if( d->m_conn_str.empty() )
  {
    throw std::runtime_error( "Database connection string (conn_str) is required" );
  }

  // Open database connection
  d->m_conn.open( d->m_conn_str );

  // Start transaction for efficient batch inserts
  d->m_transaction.reset( new ::cppdb::transaction( d->m_conn ) );
  d->m_in_transaction = true;
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_db_process
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

  // Handle periodic commits
  if( d->m_commit_interval > 0 )
  {
    d->m_commit_counter++;
    if( d->m_commit_counter >= d->m_commit_interval )
    {
      if( d->m_in_transaction && d->m_transaction )
      {
        d->m_transaction->commit();
        d->m_transaction.reset( new ::cppdb::transaction( d->m_conn ) );
      }
      d->m_commit_counter = 0;
    }
  }

  // Pass on input descriptors and UIDs
  push_to_port_using_trait( descriptor_set, vital_descriptor_set );
  push_to_port_using_trait( string_vector, string_tuple );
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_db_process
::_finalize()
{
  // Flush any remaining buffered descriptors
  if( !d->m_descriptor_buffer.empty() )
  {
    flush_buffer();
  }

  // Final commit
  if( d->m_in_transaction && d->m_transaction )
  {
    d->m_transaction->commit();
    d->m_transaction.reset();
    d->m_in_transaction = false;
  }

  if( d->m_conn.is_open() )
  {
    d->m_conn.close();
  }
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_db_process
::flush_buffer()
{
  if( !d->m_conn.is_open() )
  {
    return;
  }

  // Prepare insert statement
  ::cppdb::statement stmt = d->m_conn.create_prepared_statement(
    "INSERT INTO DESCRIPTOR(UID, VIDEO_NAME, VECTOR_DATA, VECTOR_SIZE) "
    "VALUES(?, ?, ?, ?)" );

  // Write each buffered descriptor
  for( const auto& entry : d->m_descriptor_buffer )
  {
    const std::string& uid = entry.first;
    const std::vector< double >& desc = entry.second;

    // Serialize descriptor values to comma-separated string
    std::ostringstream ss;
    for( size_t i = 0; i < desc.size(); ++i )
    {
      if( i > 0 ) ss << ",";
      ss << desc[i];
    }

    stmt.bind( 1, uid );
    stmt.bind( 2, d->m_video_name );
    stmt.bind( 3, ss.str() );
    stmt.bind( 4, static_cast< int >( desc.size() ) );
    stmt.exec();
    stmt.reset();
  }

  d->m_frame_counter = 0;
  d->m_descriptor_buffer.clear();
}


// -------------------------------------------------------------------------------
void
ingest_descriptors_db_process
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
ingest_descriptors_db_process
::make_config()
{
  declare_config_using_trait( conn_str );
  declare_config_using_trait( video_name );
  declare_config_using_trait( max_frame_buffer );
  declare_config_using_trait( max_descriptor_buffer );
  declare_config_using_trait( commit_interval );
}

} // end namespace cppdb

} // end namespace viame
