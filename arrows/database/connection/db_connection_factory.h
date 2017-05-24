/*ckwg +5
 * Copyright 2010-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef __VIDTK_DB_CONNECTION_FACTORY_H__
#define __VIDTK_DB_CONNECTION_FACTORY_H__

#include <database/connection/db_connection.h>

namespace vidtk {

#define ERR_NONE 0
#define ERR_NO_BACKEND_SUPPORT 1
#define ERR_VERSION_UPDATED    2

#define SQLITE3 "sqlite3"
#define POSTGRESQL "postgresql"
#define FILE_CONN "file"

class db_connection_factory
{

public:

  static int get_connection(
    const std::string conn_type,
    const std::string db_name,
    const std::string db_user,
    const std::string db_pass,
    const std::string db_host,
    const std::string db_port,
    db_connection_sptr & conn );

  ///connection_args is assumed to be a valid list of
  ///the form k1=v1;k2=v2; which is passed directly to the
  ///backend connection
  static int get_connection(
    const std::string conn_type,
    const std::string db_name,
    const std::string db_user,
    const std::string db_pass,
    const std::string db_host,
    const std::string db_port,
    std::string connection_args,
    db_connection_sptr & conn );

  // Get memory-only connection, in which tracks are set explicitly
  static int get_memory_connection(db_connection_sptr & conn);

private:

  db_connection_factory();

};

} // namespace vidtk

#endif //#ifndef __VIDTK_DB_CONNECTION_FACTORY_H__
