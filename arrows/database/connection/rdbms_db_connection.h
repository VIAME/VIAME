/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef __VIDTK_RDBMS_DB_CONNECTION_H__
#define __VIDTK_RDBMS_DB_CONNECTION_H__

/**
\file
\brief
Method and field definitions for a postgresql db connection.
*/

#include "db_connection.h"

#include <database/track_db.h>
#include <database/event_db.h>
#include <database/activity_db.h>
#include <database/frame_metadata_db.h>
#include <database/session_db.h>
#include <database/mission_db.h>
#include <database/detection_db.h>
#include <database/pod/pod_db.h>

#include <cppdb/frontend.h>

namespace vidtk {

#define EMPTY_SESSION_ID 0

class rdbms_db_connection
  : public db_connection
{

public:

  ///Connects to the database and tests the connection using is_connection_open
  virtual bool connect();
  ///Closes the underlying database connection
  virtual void close_connection() = 0;
  ///Tests the actual connection to the database using a small query
  virtual bool is_connection_open() = 0;
  ///Sets the currently active db_session.  Tests session validity
  virtual int set_active_session( db_session_sptr & session );
  ///Sets the active session based on the session_id.  Resolves full session and tests.
  virtual int set_active_session( vxl_int_32 session_id );
  ///Gets the current IDE version from the database
  virtual std::string get_ide_version();

  ///Gets the track_db instance
  virtual track_db_sptr get_track_db();
  ///Gets the mission_db instance
  virtual mission_db_sptr get_mission_db();
  ///Gets the session_db instance
  virtual session_db_sptr get_session_db();
  ///Gets the pod_db instance
  virtual pod_db_sptr get_pod_db();
  ///Gets the frame_metadata_db
  virtual frame_metadata_db_sptr get_frame_metadata_db();

protected:

  cppdb::session db_conn_;
  db_session_sptr current_session_;
  track_db_sptr track_db_instance_;
  mission_db_sptr mission_db_instance_;
  session_db_sptr session_db_instance_;
  pod_db_sptr pod_db_instance_;
  frame_metadata_db_sptr frame_db_instance_;
  query_manager_sptr query_mgr_;
  std::string connection_args_;

#ifndef DATABASE_API_TRACKING_ONLY
public:
  ///Gets the event_db instance
  virtual event_db_sptr get_event_db();
  ///Gets the activity_db instance
  virtual activity_db_sptr get_activity_db();

protected:
  event_db_sptr event_db_instance_;
  activity_db_sptr activity_db_instance_;

#endif


};

} //namespace vidtk

#endif //__VIDTK_RDBMS_DB_CONNECTION_H__
