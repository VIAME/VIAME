/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Attach descriptors to object track states from database
 */

#include "object_track_descriptors_db_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor.h>
#include <vital/types/object_track_set.h>

#include <cppdb/frontend.h>

#include <iostream>
#include <sstream>
#include <vector>

namespace viame
{

namespace cppdb
{

create_config_trait( conn_str, std::string, "",
  "Database connection string (e.g., postgresql:host=localhost;dbname=viame;user=postgres)" );
create_config_trait( video_name, std::string, "",
  "Video name for looking up descriptors" );

//--------------------------------------------------------------------------------
// Private implementation class
class object_track_descriptors_db_process::priv
{
public:
  priv()
    : m_conn_str( "" )
    , m_video_name( "" )
    , m_current_idx( 0 ) {}

  ~priv()
  {
    if( m_conn.is_open() )
    {
      m_conn.close();
    }
  }

  void connect_on_demand()
  {
    if( !m_conn.is_open() )
    {
      m_conn.open( m_conn_str );
    }
  }

  std::string m_conn_str;
  std::string m_video_name;
  kwiver::vital::frame_id_t m_current_idx;
  ::cppdb::session m_conn;
};

// ===============================================================================

object_track_descriptors_db_process
::object_track_descriptors_db_process( config_block_sptr const& config )
  : process( config ),
    d( new object_track_descriptors_db_process::priv() )
{
  make_ports();
  make_config();
}


object_track_descriptors_db_process
::~object_track_descriptors_db_process()
{
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_db_process
::_configure()
{
  d->m_conn_str = config_value_using_trait( conn_str );
  d->m_video_name = config_value_using_trait( video_name );

  if( d->m_conn_str.empty() )
  {
    throw std::runtime_error( "Database connection string (conn_str) is required" );
  }

  // Open database connection
  d->m_conn.open( d->m_conn_str );
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_db_process
::_step()
{
  d->connect_on_demand();

  // Grab input object tracks
  kwiver::vital::object_track_set_sptr object_tracks =
    grab_from_port_using_trait( object_track_set );

  // Prepare query statement - same query as Python smqtk_object_track_descriptors
  ::cppdb::statement uid_stmt = d->m_conn.create_prepared_statement(
    "SELECT track_descriptor.uid FROM track_descriptor "
    "INNER JOIN track_descriptor_track ON track_descriptor.uid = track_descriptor_track.uid "
    "INNER JOIN track_descriptor_history ON track_descriptor.uid = track_descriptor_history.uid "
    "WHERE track_descriptor.video_name = ? "
    "AND track_descriptor_history.frame_number = ? "
    "AND track_descriptor_track.track_id = ?" );

  ::cppdb::statement desc_stmt = d->m_conn.create_prepared_statement(
    "SELECT VECTOR_DATA FROM DESCRIPTOR WHERE UID = ?" );

  // Iterate through all tracks and states
  for( auto track : object_tracks->tracks() )
  {
    for( auto state : *track | kwiver::vital::as_object_track )
    {
      if( !state )
      {
        continue;
      }

      // Only process states at the current frame index (like Python version)
      if( state->frame() != d->m_current_idx )
      {
        continue;
      }

      kwiver::vital::track_id_t track_id = track->id();
      kwiver::vital::frame_id_t frame_id = state->frame();

      // Query for UID
      uid_stmt.bind( 1, d->m_video_name );
      uid_stmt.bind( 2, frame_id );
      uid_stmt.bind( 3, track_id );

      ::cppdb::result uid_row = uid_stmt.query();

      if( !uid_row.next() )
      {
        uid_stmt.reset();
        throw std::runtime_error( "Could not get track descriptor for track " +
          std::to_string( track_id ) + " frame " + std::to_string( frame_id ) );
      }

      std::string uid;
      uid_row.fetch( 0, uid );
      uid_stmt.reset();

      // Query for descriptor values
      desc_stmt.bind( 1, uid );
      ::cppdb::result desc_row = desc_stmt.query();

      if( desc_row.next() )
      {
        std::string vector_str;
        if( desc_row.fetch( 0, vector_str ) )
        {
          // Parse comma-separated values
          std::vector< double > values;
          std::istringstream ss( vector_str );
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
            // Create descriptor and attach to detection
            auto desc = std::make_shared<
              kwiver::vital::descriptor_dynamic< double > >( values.size() );

            double* raw = desc->raw_data();
            for( size_t i = 0; i < values.size(); ++i )
            {
              raw[i] = values[i];
            }

            kwiver::vital::detected_object_sptr detection = state->detection();
            if( detection )
            {
              detection->set_descriptor( desc );
            }

            std::cout << "Finished track state: " << track_id << " " << frame_id << std::endl;
          }
        }
      }

      desc_stmt.reset();
    }
  }

  // Increment frame index (like Python version)
  d->m_current_idx++;

  // Pass through the modified track set
  push_to_port_using_trait( object_track_set, object_tracks );
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_db_process
::make_ports()
{
  sprokit::process::port_flags_t optional;

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( object_track_set, required );

  // -- outputs --
  declare_output_port_using_trait( object_track_set, optional );
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_db_process
::make_config()
{
  declare_config_using_trait( conn_str );
  declare_config_using_trait( video_name );
}

} // end namespace cppdb

} // end namespace viame
