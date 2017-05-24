/*ckwg +5
 * Copyright 2010-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <database/connection/db_connection_factory.h>
#include <database/connection/file_db_connection.h>

#ifdef HAS_CPPDB
#include <cppdb/frontend.h>
#include <database/connection/rdbms_db_connection.h>
#include <database/connection/sqlite_db_connection.h>
#include <database/connection/postgresql_db_connection.h>
#endif


namespace vidtk
{

int
db_connection_factory
::get_connection(
  const std::string conn_type,
  const std::string db_name,
  const std::string db_user,
  const std::string db_pass,
  const std::string db_host,
  const std::string db_port,
  db_connection_sptr & conn )

{
  return get_connection( conn_type, db_name, db_user, db_pass, db_host, db_port, "", conn);
}

int
db_connection_factory
::get_connection(
  const std::string conn_type,
  const std::string db_name,
  const std::string db_user,
  const std::string db_pass,
  const std::string db_host,
  const std::string db_port,
  std::string connection_args,
  db_connection_sptr & conn )
{
  int ret = ERR_NONE;

  if ( conn_type == SQLITE3 )
  {
    #ifdef HAS_CPPDB

    conn = new sqlite_db_connection( db_name );
    #else
    (void)db_name;
    ret = ERR_NO_BACKEND_SUPPORT;
    #endif
  }

  else if ( conn_type == POSTGRESQL )
  {
    #ifdef HAS_CPPDB
    conn = new postgresql_db_connection(
      db_name, db_user, db_pass, db_host, db_port, connection_args );
    #else
    (void)db_user;
    (void)db_pass;
    (void)db_host;
    (void)db_port;
    (void)connection_args;
    ret = ERR_NO_BACKEND_SUPPORT;
    #endif
  }
  else if ( conn_type == FILE_CONN )
  {
    conn = new file_db_connection( db_name, db_user );
  }
  else
  {
    ret = ERR_NO_BACKEND_SUPPORT;
  }

  return ret;
}

int
db_connection_factory
::get_memory_connection(db_connection_sptr & conn)
{
  conn = new file_db_connection("", "");
  return ERR_NONE;
}

} // namespace vidtk
