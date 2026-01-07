/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Attach descriptors to object track states from file
 */

#include "object_track_descriptors_process.h"

#include <vital/vital_types.h>
#include <vital/types/descriptor.h>
#include <vital/types/object_track_set.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <vector>

namespace viame
{

namespace core
{

create_config_trait( descriptor_file, std::string, "track_descriptors.csv",
  "Path to the CSV file containing descriptors indexed by track_id and frame_id" );
create_config_trait( uid_descriptor_file, std::string, "",
  "Optional path to UID-indexed descriptor file (format: uid,val1,val2,...). "
  "If provided, also requires uid_mapping_file." );
create_config_trait( uid_mapping_file, std::string, "",
  "Optional path to UID mapping file (format: track_id,frame_id,uid). "
  "Used with uid_descriptor_file for UID-based lookup." );

//--------------------------------------------------------------------------------
// Private implementation class
class object_track_descriptors_process::priv
{
public:
  priv()
    : m_descriptor_file( "track_descriptors.csv" )
    , m_uid_descriptor_file( "" )
    , m_uid_mapping_file( "" )
    , m_index_loaded( false ) {}

  ~priv() {}

  std::string m_descriptor_file;
  std::string m_uid_descriptor_file;
  std::string m_uid_mapping_file;
  bool m_index_loaded;

  // Map from (track_id, frame_id) to descriptor vector
  using key_type = std::pair< kwiver::vital::track_id_t, kwiver::vital::frame_id_t >;
  std::map< key_type, std::vector< double > > m_descriptor_index;
};

// ===============================================================================

object_track_descriptors_process
::object_track_descriptors_process( config_block_sptr const& config )
  : process( config ),
    d( new object_track_descriptors_process::priv() )
{
  make_ports();
  make_config();
}


object_track_descriptors_process
::~object_track_descriptors_process()
{
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_process
::_configure()
{
  d->m_descriptor_file = config_value_using_trait( descriptor_file );
  d->m_uid_descriptor_file = config_value_using_trait( uid_descriptor_file );
  d->m_uid_mapping_file = config_value_using_trait( uid_mapping_file );

  load_descriptor_index();
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_process
::load_descriptor_index()
{
  // Check if we should use UID-based lookup
  if( !d->m_uid_descriptor_file.empty() && !d->m_uid_mapping_file.empty() )
  {
    // Load UID to descriptor mapping
    std::unordered_map< std::string, std::vector< double > > uid_to_desc;

    std::ifstream desc_file( d->m_uid_descriptor_file );
    if( !desc_file.is_open() )
    {
      throw std::runtime_error( "Failed to open UID descriptor file: " +
        d->m_uid_descriptor_file );
    }

    std::string line;
    while( std::getline( desc_file, line ) )
    {
      if( line.empty() ) continue;

      std::istringstream ss( line );
      std::string uid;

      if( !std::getline( ss, uid, ',' ) ) continue;

      std::vector< double > values;
      std::string value_str;
      while( std::getline( ss, value_str, ',' ) )
      {
        try { values.push_back( std::stod( value_str ) ); }
        catch( const std::exception& ) {}
      }

      if( !values.empty() )
      {
        uid_to_desc[uid] = std::move( values );
      }
    }

    // Load UID mapping and build index
    std::ifstream map_file( d->m_uid_mapping_file );
    if( !map_file.is_open() )
    {
      throw std::runtime_error( "Failed to open UID mapping file: " +
        d->m_uid_mapping_file );
    }

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

        auto it = uid_to_desc.find( uid );
        if( it != uid_to_desc.end() )
        {
          d->m_descriptor_index[{ track_id, frame_id }] = it->second;
        }
      }
      catch( const std::exception& ) {}
    }
  }
  else
  {
    // Direct track_id,frame_id,values format
    std::ifstream file( d->m_descriptor_file );

    if( !file.is_open() )
    {
      throw std::runtime_error( "Failed to open descriptor file: " +
        d->m_descriptor_file );
    }

    std::string line;
    while( std::getline( file, line ) )
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
          d->m_descriptor_index[{ track_id, frame_id }] = std::move( values );
        }
        catch( const std::exception& ) {}
      }
    }
  }

  d->m_index_loaded = true;
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_process
::_step()
{
  // Grab input object tracks
  kwiver::vital::object_track_set_sptr object_tracks =
    grab_from_port_using_trait( object_track_set );

  // Iterate through all tracks and states
  for( auto track : object_tracks->tracks() )
  {
    for( auto state : *track | kwiver::vital::as_object_track )
    {
      if( !state )
      {
        continue;
      }

      kwiver::vital::track_id_t track_id = track->id();
      kwiver::vital::frame_id_t frame_id = state->frame();

      auto it = d->m_descriptor_index.find( { track_id, frame_id } );

      if( it != d->m_descriptor_index.end() )
      {
        const std::vector< double >& values = it->second;

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
      }
    }
  }

  // Pass through the modified track set
  push_to_port_using_trait( object_track_set, object_tracks );
}


// -------------------------------------------------------------------------------
void
object_track_descriptors_process
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
object_track_descriptors_process
::make_config()
{
  declare_config_using_trait( descriptor_file );
  declare_config_using_trait( uid_descriptor_file );
  declare_config_using_trait( uid_mapping_file );
}

} // end namespace core

} // end namespace viame
