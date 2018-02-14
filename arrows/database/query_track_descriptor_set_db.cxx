/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "query_track_descriptor_set_db.h"

#include <vital/types/object_track_set.h>

namespace kwiver {
namespace arrows {
namespace database {

// -------------------------------------------------------------------------------
class query_track_descriptor_set_db::priv
{
public:
  priv( query_track_descriptor_set_db* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "query_track_descriptor_set_db" ) )
  { }

  ~priv() { }

  query_track_descriptor_set_db* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  cppdb::session m_conn;
  std::string m_conn_str;
  kwiver::vital::timestamp::time_t m_config_frame_time;
};

query_track_descriptor_set_db::query_track_descriptor_set_db()
  : d( new query_track_descriptor_set_db::priv( this ) )
{
}

query_track_descriptor_set_db::~query_track_descriptor_set_db()
{
}

void query_track_descriptor_set_db::set_configuration( vital::config_block_sptr config )
{
  d->m_conn_str = config->get_value< std::string > ( "conn_str", "" );
  d->m_config_frame_time = config->get_value< double > ( "frame_time", 0.03333333 ) * 1e6;
}

bool query_track_descriptor_set_db::check_configuration( vital::config_block_sptr config ) const
{
  if( !config->has_value( "conn_str" ) )
  {
    LOG_ERROR( d->m_logger, "missing required value: conn_str" );
    return false;
  }

  return true;
}

bool query_track_descriptor_set_db::get_track_descriptor( std::string const& uid,
  query_track_descriptor_set_db::desc_tuple_t& result )
{
  connect_to_database_on_demand();

  cppdb::statement stmt = d->m_conn.create_statement( "SELECT "
    "TYPE, "
    "VIDEO_NAME "
    "FROM TRACK_DESCRIPTOR "
    "WHERE UID = ?"
  );
  stmt.bind( 1, uid );
  cppdb::result row = stmt.query();
  if( !row.next() )
  {
    return false;
  }

  std::string type;
  if( !row.fetch( 0, type ) )
  {
    return false;
  }

  vital::track_descriptor_sptr td = vital::track_descriptor::create( type );
  td->resize_descriptor( 0 );
  std::get< 1 >( result ) = td;

  td->set_uid( uid );

  std::string video_name;
  if( !row.fetch( 1, video_name ) )
  {
    return false;
  }
  std::get< 0 >( result ) = video_name;

  stmt = d->m_conn.create_statement( "SELECT "
    "TRACK_ID "
    "FROM TRACK_DESCRIPTOR_TRACK "
    "WHERE UID = ?"
  );
  stmt.bind( 1, uid );
  row = stmt.query();

  while( row.next() )
  {
    uint64_t track_id;
    row.fetch( 0, track_id );
    td->add_track_id( track_id );
  }

  stmt = d->m_conn.create_statement( "SELECT "
    "FRAME_NUMBER, "
    "IMAGE_BBOX_TL_X, "
    "IMAGE_BBOX_TL_Y, "
    "IMAGE_BBOX_BR_X, "
    "IMAGE_BBOX_BR_Y, "
    "TRACK_CONFIDENCE "
    "FROM OBJECT_TRACK "
    "WHERE TRACK_ID = ? AND VIDEO_NAME = ? "
    "ORDER BY FRAME_NUMBER"
  );
  std::vector< uint64_t > track_ids = td->get_track_ids();
  std::get< 2 >( result ).clear();
  for( uint64_t track_id : track_ids )
  {
    vital::track_sptr trk = vital::track::create();
    trk->set_id( track_id );

    stmt.bind( 1, track_id );
    stmt.bind( 2, video_name );
    row = stmt.query();
    while( row.next() )
    {
      int frame_index;
      row.fetch( 0, frame_index );

      double min_x, min_y, max_x, max_y;
      row.fetch( 1, min_x );
      row.fetch( 2, min_y );
      row.fetch( 3, max_x );
      row.fetch( 4, max_y );
      vital::bounding_box_d bbox( min_x, min_y, max_x, max_y );

      double conf;
      row.fetch( 5, conf );

      vital::detected_object_sptr det =
        std::make_shared< vital::detected_object > ( bbox, conf );
      vital::track_state_sptr ots =
        std::make_shared< vital::object_track_state > ( frame_index, det );
      trk->append( ots );

      // TODO Make this configurable - either get the history entry from the
      // tracks or from a separate table. Get it from the tracks for now.
      vital::timestamp ts( frame_index * d->m_config_frame_time, frame_index );
      td->add_history_entry( vital::track_descriptor::history_entry( ts, bbox ) );
    }

    std::get< 2 >( result ).push_back( trk );

    stmt.reset();
  }

  return true;
}

void query_track_descriptor_set_db::connect_to_database_on_demand()
{
  if( !d->m_conn.is_open() )
  {
    d->m_conn.open( d->m_conn_str );
  }
}

} } } // end namespace database
