/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Fetch descriptors from database given UIDs
 */

#include "fetch_descriptors_db_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor.h>
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
create_config_trait( fail_on_missing, bool, "false",
  "If true, throw error when a requested UID is not found. If false, skip missing UIDs." );

//--------------------------------------------------------------------------------
// Private implementation class
class fetch_descriptors_db_process::priv
{
public:
  priv()
    : m_conn_str( "" )
    , m_fail_on_missing( false ) {}

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
  bool m_fail_on_missing;
  ::cppdb::session m_conn;
};

// ===============================================================================

fetch_descriptors_db_process
::fetch_descriptors_db_process( config_block_sptr const& config )
  : process( config ),
    d( new fetch_descriptors_db_process::priv() )
{
  make_ports();
  make_config();
}


fetch_descriptors_db_process
::~fetch_descriptors_db_process()
{
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_db_process
::_configure()
{
  d->m_conn_str = config_value_using_trait( conn_str );
  d->m_fail_on_missing = config_value_using_trait( fail_on_missing );

  if( d->m_conn_str.empty() )
  {
    throw std::runtime_error( "Database connection string (conn_str) is required" );
  }

  // Open database connection
  d->m_conn.open( d->m_conn_str );
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_db_process
::_step()
{
  d->connect_on_demand();

  // Grab input UIDs
  kwiver::vital::string_vector_sptr string_tuple =
    grab_from_port_using_trait( string_vector );

  std::vector< kwiver::vital::descriptor_sptr > descriptors;

  // Prepare query statement
  ::cppdb::statement stmt = d->m_conn.create_prepared_statement(
    "SELECT VECTOR_DATA FROM DESCRIPTOR WHERE UID = ?" );

  for( const std::string& uid : *string_tuple )
  {
    stmt.bind( 1, uid );
    ::cppdb::result row = stmt.query();

    if( row.next() )
    {
      std::string vector_str;
      if( row.fetch( 0, vector_str ) )
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
          // Create a new descriptor with the values
          auto desc = std::make_shared< kwiver::vital::descriptor_dynamic< double > >(
            values.size() );

          double* raw = desc->raw_data();
          for( size_t i = 0; i < values.size(); ++i )
          {
            raw[i] = values[i];
          }

          descriptors.push_back( desc );
        }
        else if( d->m_fail_on_missing )
        {
          throw std::runtime_error( "Empty descriptor for UID: " + uid );
        }
      }
      else if( d->m_fail_on_missing )
      {
        throw std::runtime_error( "Failed to fetch descriptor data for UID: " + uid );
      }
    }
    else if( d->m_fail_on_missing )
    {
      throw std::runtime_error( "Descriptor not found for UID: " + uid );
    }

    stmt.reset();
  }

  // Create descriptor set and push to output
  kwiver::vital::descriptor_set_sptr desc_set =
    std::make_shared< kwiver::vital::simple_descriptor_set >( descriptors );

  push_to_port_using_trait( descriptor_set, desc_set );
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_db_process
::make_ports()
{
  sprokit::process::port_flags_t optional;

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( string_vector, required );

  // -- outputs --
  declare_output_port_using_trait( descriptor_set, optional );
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_db_process
::make_config()
{
  declare_config_using_trait( conn_str );
  declare_config_using_trait( fail_on_missing );
}

} // end namespace cppdb

} // end namespace viame
