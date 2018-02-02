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
  {}

  ~priv() {}

  write_track_descriptor_set_db* m_parent;
  kwiver::vital::logger_handle_t m_logger;
  cppdb::session m_conn;
  std::string m_conn_str;
  std::string m_video_name;
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
  cppdb::statement insert_td_stmt = d->m_conn.create_statement( "INSERT INTO TRACK_DESCRIPTOR("
    "UID, "
    "TYPE, "
    "VIDEO_NAME"
    ") VALUES(?, ?, ?)"
  );

  cppdb::statement insert_tdt_stmt = d->m_conn.create_statement( "INSERT INTO TRACK_DESCRIPTOR_TRACK("
    "UID, "
    "TRACK_ID"
    ") VALUES(?, ?)"
  );

  for( auto td : *set )
  {
    insert_td_stmt.bind( 1, td->get_uid().value() );
    insert_td_stmt.bind( 2, td->get_type() );
    insert_td_stmt.bind( 3, d->m_video_name );

    insert_td_stmt.exec();
    insert_td_stmt.reset();

    std::vector< uint64_t > track_ids = td->get_track_ids();
    for( uint64_t track_id : track_ids )
    {
      insert_tdt_stmt.bind( 1, td->get_uid().value() );
      insert_tdt_stmt.bind( 2, track_id );

      insert_tdt_stmt.exec();
      insert_tdt_stmt.reset();
    }
  }
}

} } } // end namespace
