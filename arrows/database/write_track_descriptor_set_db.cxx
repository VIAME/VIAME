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
 * \brief Implementation of track descriptor set db output
 */

#include "write_track_descriptor_set_db.h"

#include <cppdb/frontend.h>

#include <time.h>


namespace kwiver {
namespace arrows {
namespace database {

// -------------------------------------------------------------------------------
class write_track_descriptor_set_db::priv
{
public:
  priv( write_track_descriptor_set_db* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "write_track_descriptor_set_db" ) )
    , m_write_world_loc( false )
  {}

  ~priv() {}

  write_track_descriptor_set_db* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  cppdb::session m_conn;
  std::string m_conn_str;
  std::string m_video_name;
  bool m_write_world_loc;
};


// ===============================================================================
write_track_descriptor_set_db
::write_track_descriptor_set_db()
  : d( new write_track_descriptor_set_db::priv( this ) )
{
}


write_track_descriptor_set_db
::~write_track_descriptor_set_db()
{
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_db
::set_configuration( vital::config_block_sptr config )
{
  d->m_conn_str = config->get_value< std::string > ( "conn_str", "" );
  d->m_video_name = config->get_value< std::string > ( "video_name", "" );
  d->m_write_world_loc =
    config->get_value<bool>( "write_world_loc", d->m_write_world_loc );
}


// -------------------------------------------------------------------------------
bool
write_track_descriptor_set_db
::check_configuration(vital::config_block_sptr config) const
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
write_track_descriptor_set_db
::open(std::string const& filename)
{
  d->m_conn.open( d->m_conn_str );

  cppdb::statement delete_tdt_stmt =
    d->m_conn.create_prepared_statement( "DELETE FROM TRACK_DESCRIPTOR_TRACK "
      "WHERE UID = ?"
    );
  cppdb::statement delete_tdh_stmt =
    d->m_conn.create_prepared_statement( "DELETE FROM TRACK_DESCRIPTOR_HISTORY "
      "WHERE UID = ?"
    );

  cppdb::statement select_td_stmt = d->m_conn.create_statement( "SELECT UID "
    "FROM TRACK_DESCRIPTOR "
    "WHERE VIDEO_NAME = ?"
  );
  select_td_stmt.bind( 1, d->m_video_name );
  cppdb::result result = select_td_stmt.query();
  while( result.next() )
  {
    std::string uid;
    result.fetch( 0, uid );

    delete_tdt_stmt.bind( 1, uid );
    delete_tdt_stmt.exec();
    delete_tdt_stmt.reset();

    delete_tdh_stmt.bind( 1, uid );
    delete_tdh_stmt.exec();
    delete_tdh_stmt.reset();
  }

  cppdb::statement delete_td_stmt = d->m_conn.create_statement( "DELETE FROM TRACK_DESCRIPTOR "
    "WHERE VIDEO_NAME = ?"
  );
  delete_td_stmt.bind( 1, d->m_video_name );
  delete_td_stmt.exec();
  delete_td_stmt.reset();
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_db
::close()
{
  d->m_conn.close();
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_db
::write_set( const kwiver::vital::track_descriptor_set_sptr set )
{
  cppdb::statement insert_td_stmt = d->m_conn.create_prepared_statement( "INSERT INTO TRACK_DESCRIPTOR("
    "UID, "
    "TYPE, "
    "VIDEO_NAME"
    ") VALUES(?, ?, ?)"
  );

  cppdb::statement insert_tdt_stmt = d->m_conn.create_prepared_statement( "INSERT INTO TRACK_DESCRIPTOR_TRACK("
    "UID, "
    "TRACK_ID"
    ") VALUES(?, ?)"
  );

  cppdb::statement insert_tdh_stmt = d->m_conn.create_prepared_statement( "INSERT INTO TRACK_DESCRIPTOR_HISTORY("
    "UID, "
    "FRAME_NUMBER, "
    "TIMESTAMP, "
    "IMAGE_BBOX_TL_X, "
    "IMAGE_BBOX_TL_Y, "
    "IMAGE_BBOX_BR_X, "
    "IMAGE_BBOX_BR_Y, "
    "WORLD_BBOX_TL_X, "
    "WORLD_BBOX_TL_Y, "
    "WORLD_BBOX_BR_X, "
    "WORLD_BBOX_BR_Y"
    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
  );

  for( auto td : *set )
  {
    insert_td_stmt.bind( 1, td->get_uid().value() );
    insert_td_stmt.bind( 2, td->get_type() );
    insert_td_stmt.bind( 3, d->m_video_name );

    insert_td_stmt.exec();
    insert_td_stmt.reset();

    for( uint64_t track_id : td->get_track_ids() )
    {
      insert_tdt_stmt.bind( 1, td->get_uid().value() );
      insert_tdt_stmt.bind( 2, track_id );

      insert_tdt_stmt.exec();
      insert_tdt_stmt.reset();
    }

    for( vital::track_descriptor::history_entry h : td->get_history() )
    {
      insert_tdh_stmt.bind( 1, td->get_uid().value() );
      insert_tdh_stmt.bind( 2, h.get_timestamp().get_frame() );
      insert_tdh_stmt.bind( 3, h.get_timestamp().get_time_usec() );

      insert_tdh_stmt.bind( 4, h.get_image_location().min_x() );
      insert_tdh_stmt.bind( 5, h.get_image_location().min_y() );
      insert_tdh_stmt.bind( 6, h.get_image_location().max_x() );
      insert_tdh_stmt.bind( 7, h.get_image_location().max_y() );

      if( d->m_write_world_loc )
      {
        insert_tdh_stmt.bind( 8, h.get_world_location().min_x() );
        insert_tdh_stmt.bind( 9, h.get_world_location().min_y() );
        insert_tdh_stmt.bind( 10, h.get_world_location().max_x() );
        insert_tdh_stmt.bind( 11, h.get_world_location().max_y() );
      }
      else
      {
        insert_tdh_stmt.bind_null( 8 );
        insert_tdh_stmt.bind_null( 9 );
        insert_tdh_stmt.bind_null( 10 );
        insert_tdh_stmt.bind_null( 11 );
      }

      insert_tdh_stmt.exec();
      insert_tdh_stmt.reset();
    }
  }
}

} } } // end namespace
