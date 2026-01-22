/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of track descriptor set db output
 */

#include "write_track_descriptor_set_db.h"

#include <cppdb/frontend.h>

#include <time.h>


namespace viame {

namespace kv = kwiver::vital;

// -------------------------------------------------------------------------------
class write_track_descriptor_set_db::priv
{
public:
  priv( write_track_descriptor_set_db* parent)
    : m_parent( parent )
    , m_logger( kv::get_logger( "write_track_descriptor_set_db" ) )
    , m_write_world_loc( false )
    , m_commit_interval( 1 )
  {}

  ~priv() {}

  write_track_descriptor_set_db* m_parent;
  kv::logger_handle_t m_logger;
  cppdb::session m_conn;
  std::string m_conn_str;
  std::string m_video_name;
  bool m_write_world_loc;
  unsigned int m_commit_interval;
  unsigned int m_commit_frame_counter;
  std::unique_ptr<cppdb::transaction> m_tran;
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
::set_configuration( kv::config_block_sptr config )
{
  d->m_conn_str = config->get_value< std::string > ( "conn_str", "" );
  d->m_video_name = config->get_value< std::string > ( "video_name", "" );
  d->m_write_world_loc =
    config->get_value<bool>( "write_world_loc", d->m_write_world_loc );
  d->m_commit_interval =
    config->get_value<unsigned int>( "commit_interval", d->m_commit_interval );
}


// -------------------------------------------------------------------------------
bool
write_track_descriptor_set_db
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

  d->m_commit_frame_counter = 0;

  d->m_tran.reset( new cppdb::transaction( d->m_conn ) );
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_db
::close()
{
  d->m_tran->commit();
  d->m_tran.reset();

  d->m_conn.close();
}


// -------------------------------------------------------------------------------
void
write_track_descriptor_set_db
::write_set( const kv::track_descriptor_set_sptr set,
             const std::string& source_id )
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
    insert_td_stmt.bind( 3, ( source_id.empty() ? d->m_video_name : source_id ) );

    insert_td_stmt.exec();
    insert_td_stmt.reset();

    for( uint64_t track_id : td->get_track_ids() )
    {
      insert_tdt_stmt.bind( 1, td->get_uid().value() );
      insert_tdt_stmt.bind( 2, track_id );

      insert_tdt_stmt.exec();
      insert_tdt_stmt.reset();
    }

    for( kv::track_descriptor::history_entry h : td->get_history() )
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
