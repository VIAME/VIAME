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
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CPPDB_EXPORT read_object_track_set_db
  : public kwiver::vital::algo::read_object_track_set
{
public:
  // Registered as "db" in register_algorithms.cxx
  PLUGGABLE_IMPL(
    read_object_track_set_db,
    "Reads object tracks from a database.",
    PARAM_DEFAULT(
      conn_str, std::string,
      "Database connection string.",
      "" ),
    PARAM_DEFAULT(
      video_name, std::string,
      "Name of the video the tracks belong to.",
      "" ),
    PARAM_DEFAULT(
      batch_load, bool,
      "Load every track for the video in one query rather than per frame.",
      true ) )
  virtual ~read_object_track_set_db();

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  void open( std::string const& filename ) override;
  void close() override;
  bool read_set( kwiver::vital::object_track_set_sptr& set ) override;

private:
  void initialize() override;
  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_CPPDB_READ_OBJECT_TRACK_SET_DB_H
