/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of query_track_descriptor_set_csv
 */

#include "query_track_descriptor_set_csv.h"

#include <vital/algo/algorithm.txx>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/read_track_descriptor_set.h>
#include <vital/logger/logger.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>

#include <filesystem>
#include <map>
#include <unordered_map>

namespace kv = kwiver::vital;

namespace viame {

// -----------------------------------------------------------------------------
class query_track_descriptor_set_csv::priv
{
public:
  priv( query_track_descriptor_set_csv& parent )
    : m_parent( parent )
    , m_logger( kv::get_logger( "viame.core.query_track_descriptor_set_csv" ) )
    , m_use_tracks_for_history( false )
    , m_loaded( false )
  {}

  query_track_descriptor_set_csv& m_parent;
  kv::logger_handle_t m_logger;
  bool m_use_tracks_for_history;
  bool m_loaded;

  /// uid -> ( video name, descriptor, associated tracks )
  std::unordered_map< std::string, desc_tuple_t > m_entries;

  void load_all();
  void load_basename( std::string const& folder, std::string const& name );
};

// -----------------------------------------------------------------------------
void
query_track_descriptor_set_csv::priv
::load_all()
{
  m_entries.clear();
  m_loaded = true;

  std::string const& folder = m_parent.c_database_folder;

  if( folder.empty() || !std::filesystem::is_directory( folder ) )
  {
    LOG_ERROR( m_logger, "Index folder does not exist: " << folder );
    return;
  }

  std::vector< std::string > basenames;
  for( auto const& entry : std::filesystem::directory_iterator( folder ) )
  {
    if( !entry.is_regular_file() )
    {
      continue;
    }
    std::string const filename = entry.path().filename().string();
    std::string const& postfix = m_parent.c_index_postfix;
    if( filename.size() > postfix.size() &&
        filename.compare( filename.size() - postfix.size(),
                          postfix.size(), postfix ) == 0 )
    {
      basenames.push_back( filename.substr( 0, filename.size() - postfix.size() ) );
    }
  }

  for( auto const& name : basenames )
  {
    load_basename( folder, name );
  }

  LOG_INFO( m_logger, "Loaded " << m_entries.size() << " track descriptors from "
    << basenames.size() << " indexed video(s) in " << folder );
}

// -----------------------------------------------------------------------------
void
query_track_descriptor_set_csv::priv
::load_basename( std::string const& folder, std::string const& name )
{
  std::filesystem::path base( folder );
  std::string const desc_file =
    ( base / ( name + m_parent.c_descriptor_postfix ) ).string();
  std::string const track_file =
    ( base / ( name + m_parent.c_track_postfix ) ).string();

  if( !std::filesystem::exists( desc_file ) )
  {
    LOG_WARN( m_logger, "Indexed video " << name
      << " has no descriptor file " << desc_file << "; skipping" );
    return;
  }

  // Descriptors: the raw vectors are not needed here (the query engine
  // holds its own copy of the index), only uids, track references and history.
  kv::track_descriptor_set_sptr descs;
  try
  {
    auto reader = kv::create_algorithm< kv::algo::read_track_descriptor_set >(
      m_parent.c_descriptor_reader_type );
    auto config = reader->get_configuration();
    config->set_value( "batch_load", true );
    config->set_value( "read_raw_descriptor", false );
    reader->set_configuration( config );
    reader->open( desc_file );
    reader->read_set( descs );
    reader->close();
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( m_logger, "Unable to read " << desc_file << ": " << e.what() );
    return;
  }

  // Tracks: optional, referenced by id from the descriptors
  std::map< uint64_t, kv::track_sptr > id_to_track;
  if( std::filesystem::exists( track_file ) )
  {
    try
    {
      auto reader = kv::create_algorithm< kv::algo::read_object_track_set >(
        m_parent.c_track_reader_type );
      auto config = reader->get_configuration();
      config->set_value( "batch_load", true );
      reader->set_configuration( config );
      reader->open( track_file );
      kv::object_track_set_sptr tracks;
      reader->read_set( tracks );
      reader->close();
      if( tracks )
      {
        for( auto const& trk : tracks->tracks() )
        {
          id_to_track[ trk->id() ] = trk;
        }
      }
    }
    catch( std::exception const& e )
    {
      LOG_ERROR( m_logger, "Unable to read " << track_file << ": " << e.what() );
    }
  }

  if( !descs )
  {
    return;
  }

  for( auto const& desc : *descs )
  {
    std::vector< kv::track_sptr > assc_tracks;
    for( auto id : desc->get_track_ids() )
    {
      auto itr = id_to_track.find( id );
      if( itr != id_to_track.end() )
      {
        assc_tracks.push_back( itr->second );
      }
    }

    // Mirror the database implementation: with use_tracks_for_history the
    // history is the associated tracks' states rather than the stored one.
    if( m_use_tracks_for_history && !assc_tracks.empty() )
    {
      kv::track_descriptor::descriptor_history_t history;
      for( auto const& trk : assc_tracks )
      {
        for( auto const& state : *trk )
        {
          auto ots = std::dynamic_pointer_cast< kv::object_track_state >( state );
          if( !ots || !ots->detection() )
          {
            continue;
          }
          kv::timestamp ts( ots->time(), ots->frame() );
          history.push_back( kv::track_descriptor::history_entry(
            ts, ots->detection()->bounding_box() ) );
        }
      }
      if( !history.empty() )
      {
        desc->set_history( history );
      }
    }

    m_entries[ desc->get_uid().value() ] =
      desc_tuple_t( name, desc, assc_tracks );
  }
}

// =============================================================================
void
query_track_descriptor_set_csv
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
}

query_track_descriptor_set_csv
::~query_track_descriptor_set_csv()
{}

// -----------------------------------------------------------------------------
bool
query_track_descriptor_set_csv
::check_configuration( kv::config_block_sptr config ) const
{
  std::string const folder = config->get_value< std::string >( "database_folder", "" );
  if( folder.empty() )
  {
    LOG_ERROR( d->m_logger, "missing required value: database_folder" );
    return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
bool
query_track_descriptor_set_csv
::get_track_descriptor( std::string const& uid, desc_tuple_t& result )
{
  if( !d->m_loaded )
  {
    d->load_all();
  }

  auto itr = d->m_entries.find( uid );
  if( itr == d->m_entries.end() )
  {
    return false;
  }

  result = itr->second;
  return true;
}

// -----------------------------------------------------------------------------
void
query_track_descriptor_set_csv
::use_tracks_for_history( bool value )
{
  if( value != d->m_use_tracks_for_history )
  {
    d->m_use_tracks_for_history = value;
    // Histories are derived at load time; reload on the next lookup.
    d->m_loaded = false;
  }
}

} // end namespace viame
