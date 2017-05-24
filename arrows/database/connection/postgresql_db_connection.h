/*ckwg +5
 * Copyright 2011-2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef __VIDTK_POSTGRESQL_DB_CONNECTION_H__
#define __VIDTK_POSTGRESQL_DB_CONNECTION_H__

/**
\file
\brief
Method and field definitions for a postgresql db connection.
*/

#include "rdbms_db_connection.h"
#include <database/connection/postgresql_query_manager.h>

namespace vidtk {

class postgresql_db_connection
  : public rdbms_db_connection
{

public:

  postgresql_db_connection(
    const std::string &db_name,
    const std::string &db_user,
    const std::string &db_pass,
    const std::string &db_host,
    const std::string &db_port,
    const std::string &conn_args);

  virtual ~postgresql_db_connection();

  ///Connects to the database and tests the connection using is_connection_open
  virtual bool connect( );
  ///Closes the underlying database connection
  virtual void close_connection( );
  ///Tests the actual connection to the database using a small query
  virtual bool is_connection_open();

};

} //namespace vidtk

#endif //__VIDTK_POSTGRESQL_DB_CONNECTION_H__
