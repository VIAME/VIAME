/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for write_object_track_set_db
 */

#ifndef VIAME_CPPDB_WRITE_OBJECT_TRACK_SET_DB_H
#define VIAME_CPPDB_WRITE_OBJECT_TRACK_SET_DB_H

#include <vital/vital_config.h>
#include "viame_cppdb_export.h"

#include <vital/algo/write_object_track_set.h>

#include <memory>

namespace viame {

class VIAME_CPPDB_EXPORT write_object_track_set_db
  : public kwiver::vital::algorithm_impl< write_object_track_set_db,
      kwiver::vital::algo::write_object_track_set >
{
public:
  write_object_track_set_db();
  virtual ~write_object_track_set_db();

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();

  virtual void write_set(
    kwiver::vital::object_track_set_sptr const& set,
    kwiver::vital::timestamp const& ts,
    std::string const& file_id );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace viame

#endif // VIAME_CPPDB_WRITE_OBJECT_TRACK_SET_DB_H
