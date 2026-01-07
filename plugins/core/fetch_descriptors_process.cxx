/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Fetch descriptors from file given UIDs
 */

#include "fetch_descriptors_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor.h>
#include <vital/types/descriptor_set.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace viame
{

namespace core
{

create_config_trait( descriptor_file, std::string, "descriptors.csv",
  "Path to the CSV file containing descriptors with UIDs" );
create_config_trait( fail_on_missing, bool, "false",
  "If true, throw error when a requested UID is not found. If false, skip missing UIDs." );

//--------------------------------------------------------------------------------
// Private implementation class
class fetch_descriptors_process::priv
{
public:
  priv()
    : m_descriptor_file( "descriptors.csv" )
    , m_fail_on_missing( false )
    , m_index_loaded( false ) {}

  ~priv() {}

  std::string m_descriptor_file;
  bool m_fail_on_missing;
  bool m_index_loaded;

  // Map from UID to descriptor vector
  std::unordered_map< std::string, std::vector< double > > m_descriptor_index;
};

// ===============================================================================

fetch_descriptors_process
::fetch_descriptors_process( config_block_sptr const& config )
  : process( config ),
    d( new fetch_descriptors_process::priv() )
{
  make_ports();
  make_config();
}


fetch_descriptors_process
::~fetch_descriptors_process()
{
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_process
::_configure()
{
  d->m_descriptor_file = config_value_using_trait( descriptor_file );
  d->m_fail_on_missing = config_value_using_trait( fail_on_missing );

  load_descriptor_index();
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_process
::load_descriptor_index()
{
  std::ifstream file( d->m_descriptor_file );

  if( !file.is_open() )
  {
    throw std::runtime_error( "Failed to open descriptor file: " + d->m_descriptor_file );
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

    // First field is the UID
    if( !std::getline( ss, uid, ',' ) )
    {
      continue;
    }

    // Remaining fields are the descriptor values
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
      d->m_descriptor_index[uid] = std::move( values );
    }
  }

  d->m_index_loaded = true;
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_process
::_step()
{
  // Grab input UIDs
  kwiver::vital::string_vector_sptr string_tuple =
    grab_from_port_using_trait( string_vector );

  std::vector< kwiver::vital::descriptor_sptr > descriptors;

  for( const std::string& uid : *string_tuple )
  {
    auto it = d->m_descriptor_index.find( uid );

    if( it != d->m_descriptor_index.end() )
    {
      const std::vector< double >& values = it->second;

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
      throw std::runtime_error( "Descriptor not found for UID: " + uid );
    }
    // If not fail_on_missing, we simply skip this UID
  }

  // Create descriptor set and push to output
  kwiver::vital::descriptor_set_sptr desc_set =
    std::make_shared< kwiver::vital::simple_descriptor_set >( descriptors );

  push_to_port_using_trait( descriptor_set, desc_set );
}


// -------------------------------------------------------------------------------
void
fetch_descriptors_process
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
fetch_descriptors_process
::make_config()
{
  declare_config_using_trait( descriptor_file );
  declare_config_using_trait( fail_on_missing );
}

} // end namespace core

} // end namespace viame
