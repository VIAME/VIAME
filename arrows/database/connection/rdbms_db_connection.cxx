/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "rdbms_db_connection.h"

#include <database/connection/ide_version.h>
#include <logger/logger.h>

VIDTK_LOGGER("rdbms_db_connection_cxx");

namespace vidtk
{

bool
rdbms_db_connection
::connect()
{
  std::string current_version = this->get_ide_version();
  if ( current_version != DB_IDE_VERSION )
  {
    std::stringstream error_msg;

    error_msg
      << "Version mismatch\n"
      << "IDE_VERSION: " DB_IDE_VERSION
      << "\nDB_VERION: " << current_version
      << "\nYou need to upgrade your database before running.\n"
      "Upgrade utilities are provided for upgrading one numbered version "
      "to the next, and must be done incrementally. \nIf your DB_VERSION = "
      "0.0.0, it means you have an unversioned database to which no "
      "upgrade path is available. \nIn that case, you must\n"
      "1) Recreate your database with this IDE_VERSION.\n"
      "2) Contact the db administrator for help upgrading.\n"
      "3) Fend for yourself, figure out what's wrong and fix it.";

    LOG_ERROR( error_msg.str() );
    this->close_connection();

    return false;
  }

  return true;
}

int
rdbms_db_connection
::set_active_session( db_session_sptr & session )
{
  current_session_ = session;

  get_session_db();
  int ret = session_db_instance_->test_session( session );

  if (ret != ERR_NONE)
  {
    return ret;
  }

  //change db classes current session
  if (track_db_instance_)
  {
    track_db_instance_->set_active_session( current_session_ );
  }

#ifndef DATABASE_API_TRACKING_ONLY
  if (event_db_instance_)
  {
    event_db_instance_->set_active_session( current_session_ );
  }

  if (activity_db_instance_)
  {
    activity_db_instance_->set_active_session( current_session_ );
  }
#endif

  return ret;
}

int
rdbms_db_connection
::set_active_session( vxl_int_32 session_id )
{
  db_session_sptr session;
  get_session_db();
  bool valid = session_db_instance_->get_session_by_id( session_id, session );
  if ( !valid )
  {
    return ERR_BAD_SESSION_ID;
  }

  return set_active_session( session );
}

track_db_sptr
rdbms_db_connection
::get_track_db()
{
  if (!track_db_instance_)
  {
    track_db_instance_ = new track_db( db_conn_, query_mgr_ );
    track_db_instance_->set_active_session( current_session_ );
  }
  return track_db_instance_;
}

#ifndef DATABASE_API_TRACKING_ONLY
event_db_sptr
rdbms_db_connection
::get_event_db()
{
  if (!event_db_instance_)
  {
    event_db_instance_ = new event_db( db_conn_, get_track_db(), query_mgr_ );
    event_db_instance_->set_active_session( current_session_ );
  }
  return event_db_instance_;
}

activity_db_sptr
rdbms_db_connection
::get_activity_db()
{
  if (!activity_db_instance_)
  {
    activity_db_instance_ = new activity_db( db_conn_, get_event_db(), query_mgr_ );
    activity_db_instance_->set_active_session( current_session_ );
  }
  return activity_db_instance_;
}
#endif

frame_metadata_db_sptr
rdbms_db_connection
::get_frame_metadata_db()
{
  if (!frame_db_instance_)
  {
    frame_db_instance_ = new frame_metadata_db( db_conn_, query_mgr_ );
  }
  return frame_db_instance_;
}

mission_db_sptr
rdbms_db_connection
::get_mission_db()
{
  if (!mission_db_instance_)
  {
     mission_db_instance_ = new mission_db( db_conn_, query_mgr_ );
  }
  return mission_db_instance_;
}

session_db_sptr
rdbms_db_connection
::get_session_db()
{
  if (!session_db_instance_)
  {
     session_db_instance_ = new session_db( db_conn_, query_mgr_ );
  }
  return session_db_instance_;
}

pod_db_sptr
rdbms_db_connection
::get_pod_db()
{
  if (!pod_db_instance_)
  {
     pod_db_instance_ = new pod_db( db_conn_, query_mgr_ );
  }
  return pod_db_instance_;
}


std::string
rdbms_db_connection
::get_ide_version()
{
  std::string version = "0.0.0";

  try
  {
    std::string query = "select version from IDE_VERSION";
    cppdb::statement version_stmt = db_conn_.prepare( query );
    cppdb::result rs = version_stmt.query();

    if (rs.next())
    {
      version = rs.get<std::string>( "VERSION" );
    }
  }
  catch( cppdb::cppdb_error const &e )
  {
    LOG_ERROR( e.what() );
  }

  return version;
}

}
