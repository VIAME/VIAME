/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef __POSTGRESQL_QUERY_MANAGER_H__
#define __POSTGRESQL_QUERY_MANAGER_H__

#include "query_manager.h"

namespace vidtk {


class postgresql_query_manager
  : public query_manager
{

public:

  postgresql_query_manager();
  ~postgresql_query_manager();
  const std::string & insert_track_query() const;
  const std::string & insert_track_state_query() const;

protected:

  static const std::string POSTGIS_LINESTRING_INSERT_STMT;
  static const std::string POSTGIS_POINT_INSERT_STMT;
  static const std::string POSTGIS_POLYGON_INSERT_STMT;

private:


}; //class postgresql_query_manager

}// namespace vidtk


#endif// __POSTGRESQL_QUERY_MANAGER_H__
