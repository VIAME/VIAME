/*ckwg +5
 * Copyright 2011-2015 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "postgresql_db_connection.h"
#include <logger/logger.h>

VIDTK_LOGGER("postgresql_db_connection_cxx");

namespace vidtk {

postgresql_db_connection
::postgresql_db_connection(
  const std::string &db_name,
  const std::string &db_user,
  const std::string &db_pass,
  const std::string &db_host,
  const std::string &db_port,
  const std::string &conn_args)
{
  connect_string_ += ( "postgresql:" );
#ifdef MODULE_PATH
  connect_string_ += "@modules_path="  MODULE_PATH;
#endif
  connect_string_ += ( ";@blob=bytea");
  connect_string_ += ( ";host=" + db_host);
  connect_string_ += ( ";user=" + db_user);
  connect_string_ += ( ";password=" + db_pass);
  connect_string_ += ( ";dbname=" + db_name);
  connect_string_ += ( ";port=" + db_port);

  connection_args_ = conn_args;

  if ( connection_args_.length() > 0 )
  {
    connect_string_ += ";" + connection_args_;
  }

  connect_called_ = false;
}

postgresql_db_connection
::~postgresql_db_connection()
{
  if (db_conn_.is_open() )
  {
    db_conn_.close();
  }
}


bool
postgresql_db_connection
::connect( )
{
  try
  {
    if (connect_string_ != "")
    {
      db_conn_.open( connect_string_ );
      connect_called_ = true;
      query_mgr_ = new postgresql_query_manager();

      if( is_connection_open() )
      {
        return rdbms_db_connection::connect();
      }
    }
  }
  catch(cppdb::cppdb_error const &e)
  {
    LOG_ERROR("Failed to make database connection " << e.what());
    return false;
  }

  return false;
}

void
postgresql_db_connection
::close_connection( )
{
  if (db_conn_.is_open() )
  {
    db_conn_.close();
  }
}


bool
postgresql_db_connection
::is_connection_open( )
{
  if ( !connect_called_ )
  {
    return false;
  }

  return db_conn_.is_open();
}

}
