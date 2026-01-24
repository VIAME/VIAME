/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_object_track_set_dive
 */

#include "read_object_track_set_dive.h"
#include "read_detected_object_set_dive.h"  // For shared DIVE parsing functions

#include <vital/types/object_track_set.h>
#include <vital/util/data_stream_reader.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <memory>
#include <fstream>


namespace viame {

// -----------------------------------------------------------------------------------
class read_object_track_set_dive::priv
{
public:
  priv( read_object_track_set_dive& parent )
    : m_parent( &parent )
    , m_first( true )
  { }

  ~priv() { }

  void read_all();

  read_object_track_set_dive* m_parent;
  bool m_first;

  std::string m_filename;

  // All tracks loaded from the file
  std::vector< kwiver::vital::track_sptr > m_all_tracks;

  // Tracks indexed by frame for streaming mode
  std::map< int, std::vector< kwiver::vital::track_sptr > > m_tracks_by_frame;

  int m_current_frame;
  int m_max_frame;
};


// ===================================================================================
read_object_track_set_dive
::~read_object_track_set_dive()
{
}


// -----------------------------------------------------------------------------------
void
read_object_track_set_dive
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.read_object_track_set_dive" );
}


// -----------------------------------------------------------------------------------
bool
read_object_track_set_dive
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
void
read_object_track_set_dive
::open( std::string const& filename )
{
  kwiver::vital::algo::read_object_track_set::open( filename );

  d->m_first = true;
  d->m_filename = filename;
  d->m_all_tracks.clear();
  d->m_tracks_by_frame.clear();
  d->m_current_frame = 0;
  d->m_max_frame = -1;
}


// -----------------------------------------------------------------------------------
bool
read_object_track_set_dive
::read_set( kwiver::vital::object_track_set_sptr& set )
{
  if( d->m_first )
  {
    d->read_all();
    d->m_first = false;
  }

  if( true )
  {
    // Return all tracks in one set
    if( d->m_all_tracks.empty() )
    {
      set = std::make_shared< kwiver::vital::object_track_set >();
      return false;
    }

    set = std::make_shared< kwiver::vital::object_track_set >( d->m_all_tracks );
    d->m_all_tracks.clear();
    return true;
  }
  else
  {
    // Streaming mode - return tracks for current frame
    if( d->m_current_frame > d->m_max_frame )
    {
      set = std::make_shared< kwiver::vital::object_track_set >();
      return false;
    }

    auto itr = d->m_tracks_by_frame.find( d->m_current_frame );
    if( itr != d->m_tracks_by_frame.end() )
    {
      set = std::make_shared< kwiver::vital::object_track_set >( itr->second );
    }
    else
    {
      set = std::make_shared< kwiver::vital::object_track_set >();
    }

    ++d->m_current_frame;
    return true;
  }
}


// ===================================================================================
void
read_object_track_set_dive::priv
::read_all()
{
  m_all_tracks.clear();
  m_tracks_by_frame.clear();
  m_current_frame = 0;
  m_max_frame = -1;

  // Read the JSON filename from the stream (first non-empty, non-comment line)
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  while( stream_reader.getline( line ) )
  {
    // Trim whitespace
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos )
    {
      continue; // Skip empty lines
    }
    if( line[start] == '#' )
    {
      continue; // Skip comments
    }

    size_t end = line.find_last_not_of( " \t\r\n" );
    std::string json_file = line.substr( start, end - start + 1 );

    // Parse the JSON file using shared function from detection reader
    dive_annotation_file dive_data;
    if( !parse_dive_json_file( json_file, m_parent->logger(), dive_data ) )
    {
      LOG_ERROR( m_parent->logger(),
                 "Failed to parse DIVE JSON file: " << json_file );
      continue;
    }

    // Process each DIVE track into a kwiver track
    for( auto const& track_pair : dive_data.tracks )
    {
      dive_track const& dtrack = track_pair.second;

      // Create a new kwiver track
      auto track = kwiver::vital::track::create();
      track->set_id( dtrack.id );

      // Process each feature in the DIVE track
      for( dive_feature const& feature : dtrack.features )
      {
        int frame = feature.frame;

        // Update max frame
        if( frame > m_max_frame )
        {
          m_max_frame = frame;
        }

        // Create detection using shared function
        auto det = create_detected_object_from_dive( feature, dtrack.confidencePairs );
        if( !det )
        {
          continue;
        }

        // Create track state with frame number and detection
        kwiver::vital::time_usec_t frame_time = frame;
        auto state = std::make_shared< kwiver::vital::object_track_state >(
          frame, frame_time, det );

        track->append( state );

        // Index track by frame for streaming mode
        if( false )
        {
          m_tracks_by_frame[ frame ].push_back( track );
        }
      }

      // Only add track if it has states
      if( track->size() > 0 )
      {
        m_all_tracks.push_back( track );
      }
    }
  }

  LOG_DEBUG( m_parent->logger(),
             "Loaded " << m_all_tracks.size() << " tracks from DIVE JSON" );
}

} // end namespace viame
