/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of detected object set csv output
 */

#include "write_object_track_set_db.h"

#include <cppdb/frontend.h>

#include <time.h>

namespace viame {

namespace kv = kwiver::vital;

// -------------------------------------------------------------------------------
class write_object_track_set_db::priv
{
public:
  priv( write_object_track_set_db* parent)
    : m_parent( parent )
    , m_logger( kv::get_logger( "write_object_track_set_db" ) )
    , m_commit_interval( 1 )
    , m_matching_frames_only( true )
  { }

  ~priv() { }

  write_object_track_set_db* m_parent;
  kv::logger_handle_t m_logger;
  cppdb::session m_conn;
  std::string m_conn_str;
  std::string m_video_name;
  unsigned int m_commit_interval;
  bool m_matching_frames_only;
  unsigned int m_commit_frame_counter;
  std::unique_ptr<cppdb::transaction> m_tran;
};


// ===============================================================================
write_object_track_set_db
::write_object_track_set_db()
  : d( new write_object_track_set_db::priv( this ) )
{
}


write_object_track_set_db
::~write_object_track_set_db()
{
}


// -------------------------------------------------------------------------------
void
write_object_track_set_db
::set_configuration(kv::config_block_sptr config)
{
  d->m_conn_str = config->get_value< std::string > ( "conn_str", "" );
  d->m_video_name = config->get_value< std::string > ( "video_name", "" );
  d->m_commit_interval =
    config->get_value<unsigned int>( "commit_interval", d->m_commit_interval );
  d->m_matching_frames_only =
    config->get_value<bool>( "matching_frames_only", d->m_matching_frames_only );
}


// -------------------------------------------------------------------------------
bool
write_object_track_set_db
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
write_object_track_set_db
::open(std::string const& filename)
{
  d->m_conn.open( d->m_conn_str );

  cppdb::statement delete_stmt = d->m_conn.create_statement( "DELETE FROM OBJECT_TRACK "
    "WHERE VIDEO_NAME = ?"
  );
  delete_stmt.bind( 1, d->m_video_name );
  delete_stmt.exec();

  d->m_commit_frame_counter = 0;

  d->m_tran.reset( new cppdb::transaction( d->m_conn ) );
}


// -------------------------------------------------------------------------------
void
write_object_track_set_db
::close()
{
  d->m_tran->commit();
  d->m_tran.reset();

  d->m_conn.close();
}


// -------------------------------------------------------------------------------
void
write_object_track_set_db
::write_set( kv::object_track_set_sptr const& set,
             kv::timestamp const& ts,
             std::string const & file_id )
{
  cppdb::statement update_stmt = d->m_conn.create_prepared_statement( "UPDATE OBJECT_TRACK SET "
    "TIMESTAMP = ?, "
    "IMAGE_BBOX_TL_X = ?, "
    "IMAGE_BBOX_TL_Y = ?, "
    "IMAGE_BBOX_BR_X = ?, "
    "IMAGE_BBOX_BR_Y = ?, "
    "TRACK_CONFIDENCE = ? "
    "WHERE TRACK_ID = ? AND FRAME_NUMBER = ? AND VIDEO_NAME = ?"
  );

  cppdb::statement insert_stmt = d->m_conn.create_prepared_statement( "INSERT INTO OBJECT_TRACK("
    "TRACK_ID, "
    "FRAME_NUMBER, "
    "VIDEO_NAME, "
    "TIMESTAMP, "
    "IMAGE_BBOX_TL_X, "
    "IMAGE_BBOX_TL_Y, "
    "IMAGE_BBOX_BR_X, "
    "IMAGE_BBOX_BR_Y, "
    "TRACK_CONFIDENCE"
    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
  );

  for( auto trk : set->tracks() )
  {
    for( auto ts_ptr : *trk )
    {
      kv::object_track_state* trkstate =
        dynamic_cast< kv::object_track_state* >( ts_ptr.get() );

      if( !trkstate )
      {
        LOG_ERROR( d->m_logger, "MISSED STATE " << trk->id() << " " << trk->size() );
        continue;
      }

      if( d->m_matching_frames_only && trkstate->frame() != ts.get_frame() )
      {
        continue;
      }

      kv::detected_object_sptr det = trkstate->detection();
      const kv::bounding_box_d empty_box = kv::bounding_box_d( -1, -1, -1, -1 );
      kv::bounding_box_d bbox = ( det ? det->bounding_box() : empty_box );

      update_stmt.bind( 1, trkstate->time() );
      update_stmt.bind( 2, bbox.min_x() );
      update_stmt.bind( 3, bbox.min_y() );
      update_stmt.bind( 4, bbox.max_x() );
      update_stmt.bind( 5, bbox.max_y() );
      update_stmt.bind( 6, det->confidence() );
      update_stmt.bind( 7, trk->id() );
      update_stmt.bind( 8, trkstate->frame() );
      update_stmt.bind( 9, d->m_video_name );

      update_stmt.exec();
      unsigned long long count = update_stmt.affected();
      update_stmt.reset();

      if( count == 0 )
      {
        insert_stmt.bind( 1, trk->id() );
        insert_stmt.bind( 2, trkstate->frame() );
        insert_stmt.bind( 3, d->m_video_name );
        insert_stmt.bind( 4, trkstate->time() );
        insert_stmt.bind( 5, bbox.min_x() );
        insert_stmt.bind( 6, bbox.min_y() );
        insert_stmt.bind( 7, bbox.max_x() );
        insert_stmt.bind( 8, bbox.max_y() );
        insert_stmt.bind( 9, det->confidence() );

        insert_stmt.exec();
        insert_stmt.reset();
      }
    }
  }

  if( d->m_commit_interval > 0 )
  {
    d->m_commit_frame_counter++;

    if( d->m_commit_frame_counter >= d->m_commit_interval )
    {
      d->m_tran->commit();
      d->m_tran.reset( new cppdb::transaction( d->m_conn ) );
      d->m_commit_frame_counter = 0;
    }
  }
}

} // end namespace viame
