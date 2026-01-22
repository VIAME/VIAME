/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_db
 */

#ifndef VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H
#define VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H

#include <vital/vital_config.h>
#include "viame_cppdb_export.h"

#include <vital/algo/read_object_track_set.h>

#include <memory>

namespace viame {

class VIAME_CPPDB_EXPORT read_object_track_set_db
  : public kwiver::vital::algorithm_impl< read_object_track_set_db,
      kwiver::vital::algo::read_object_track_set >
{
public:
  read_object_track_set_db();
  virtual ~read_object_track_set_db();

  void set_configuration( kwiver::vital::config_block_sptr config ) override;
  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  void open( std::string const& filename ) override;
  void close() override;
  bool read_set( kwiver::vital::object_track_set_sptr& set ) override;

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace viame

#endif // VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H
