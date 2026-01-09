/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation of object track set database input
 */

#include "read_object_track_set_db.h"

#include <cppdb/frontend.h>

namespace viame {

namespace kv = kwiver::vital;

// -------------------------------------------------------------------------------
class read_object_track_set_db::priv
{
public:
  priv( read_object_track_set_db* parent)
    : m_parent( parent )
    , m_logger( kv::get_logger( "read_object_track_set_db" ) )
    , m_first( true )
    , m_batch_load( true )
    , m_current_idx( 0 )
    , m_last_idx( 1 )
  { }

  ~priv() { }

  read_object_track_set_db* m_parent;
  kv::logger_handle_t m_logger;
  bool m_first;
  bool m_batch_load;
  cppdb::session m_conn;
  std::string m_conn_str;
  std::string m_video_name;

  kv::frame_id_t m_current_idx;
  kv::frame_id_t m_last_idx;

  void read_all();

  // Map of object tracks indexed by frame number. Each set contains all tracks
  // referenced (active) on that individual frame.
  std::map< kv::frame_id_t, std::vector< kv::track_sptr > > m_tracks_by_frame_id;

  // Compilation of all loaded tracks, track id -> track sptr mapping
  std::map< kv::frame_id_t, kv::track_sptr > m_all_tracks;

  // Compilation of all loaded track IDs, track id -> type string
  std::map< kv::frame_id_t, std::string > m_track_ids;
};


// ===============================================================================
read_object_track_set_db
::read_object_track_set_db()
  : d( new read_object_track_set_db::priv( this ) )
{
}


read_object_track_set_db
::~read_object_track_set_db()
{
}


// -------------------------------------------------------------------------------
void
read_object_track_set_db
::set_configuration(kv::config_block_sptr config)
{
  d->m_conn_str = config->get_value< std::string > ( "conn_str", "" );
  d->m_video_name = config->get_value< std::string > ( "video_name", "" );
  d->m_batch_load = config->get_value<bool>( "batch_load", d->m_batch_load );
}


// -------------------------------------------------------------------------------
bool
read_object_track_set_db
::check_configuration(kv::config_block_sptr config) const
{
  if( !config->has_value( "conn_str" ) )
  {
    LOG_ERROR( d->m_logger, "missing required value: conn_str" );
    return false;
  }

  if( !config->has_value( "video_name" ) )
  {
    LOG_ERROR( d->m_logger, "missing required value: video_name" );
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------
void
read_object_track_set_db
::open(std::string const& filename)
{
  d->m_conn.open( d->m_conn_str );
}


// -------------------------------------------------------------------------------
void
read_object_track_set_db
::close()
{
  d->m_conn.close();
}


// -------------------------------------------------------------------------------
bool
read_object_track_set_db
::read_set( kv::object_track_set_sptr& set )
{
  auto const first = d->m_first;
  if( first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;
  }

  if( d->m_batch_load )
  {
    if( !first )
    {
      return false;
    }
    std::vector< kv::track_sptr > trks;

    for( auto it = d->m_all_tracks.begin(); it != d->m_all_tracks.end(); ++it )
    {
      trks.push_back( it->second );
    }

    set = kv::object_track_set_sptr( new kv::object_track_set( trks ) );
    return true;
  }

  if( d->m_current_idx > d->m_last_idx )
  {
    return false;
  }

  // Return detection set at current index if there is one
  if( d->m_tracks_by_frame_id.count( d->m_current_idx ) == 0 )
  {
    // Return empty set
    set = std::make_shared< kv::object_track_set >();
  }
  else
  {
    // Return tracks for this frame
    set = std::make_shared< kv::object_track_set >( d->m_tracks_by_frame_id[ d->m_current_idx ] );
  }

  ++d->m_current_idx;

  return true;
}


// -------------------------------------------------------------------------------
void
read_object_track_set_db::priv
::read_all()
{
  m_tracks_by_frame_id.clear();
  m_all_tracks.clear();
  m_track_ids.clear();

  cppdb::statement stmt = m_conn.create_statement( "SELECT "
    "TRACK_ID, "
    "FRAME_NUMBER, "
    "TIMESTAMP, "
    "IMAGE_BBOX_TL_X, "
    "IMAGE_BBOX_TL_Y, "
    "IMAGE_BBOX_BR_X, "
    "IMAGE_BBOX_BR_Y, "
    "TRACK_CONFIDENCE "
    "FROM OBJECT_TRACK "
    "WHERE VIDEO_NAME = ?"
  );
  stmt.bind( 1, m_video_name );

  cppdb::result row = stmt.query();

  while( row.next() )
  {
    /*
     * Check to see if we have seen this frame before. If we have,
     * then retrieve the frame's index into our output map. If not
     * seen before, add frame -> detection set index to our map and
     * press on.
     *
     * This allows for track states to be written in a non-contiguous
     * manner as may be done by streaming writers.
     */
    kv::frame_id_t frame_index;
    kv::time_usec_t frame_time;
    int track_index;

    row.fetch( 0, track_index );
    row.fetch( 1, frame_index );
    row.fetch( 2, frame_time );

    double min_x, min_y, max_x, max_y;

    row.fetch( 3, min_x );
    row.fetch( 4, min_y );
    row.fetch( 5, max_x );
    row.fetch( 6, max_y );

    kv::bounding_box_d bbox(
      min_x,
      min_y,
      max_x,
      max_y );

    double conf;
    row.fetch( 7, conf );

    // Create new detection
    kv::detected_object_sptr det =
      std::make_shared< kv::detected_object >( bbox, conf );

    // Create new object track state
    kv::track_state_sptr ots =
      std::make_shared< kv::object_track_state >( frame_index, frame_time, det );

    // Assign object track state to track
    kv::track_sptr trk;

    if( m_all_tracks.count( track_index ) == 0 )
    {
      trk = kv::track::create();
      trk->set_id( track_index );
      m_all_tracks[ track_index ] = trk;
    }
    else
    {
      trk = m_all_tracks[ track_index ];
    }

    trk->append( ots );

    // Add track to indexes
    if( !m_batch_load )
    {
      m_tracks_by_frame_id[ frame_index ].push_back( trk );
      m_last_idx = std::max( m_last_idx, frame_index );
    }
  }
}

} // end namespace viame
