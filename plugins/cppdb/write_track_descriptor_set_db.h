/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for write_track_descriptor_set_db
 */

#ifndef VIAME_CPPDB_WRITE_TRACK_DESCRIPTOR_SET_DB_H
#define VIAME_CPPDB_WRITE_TRACK_DESCRIPTOR_SET_DB_H

#include <vital/vital_config.h>
#include "viame_cppdb_export.h"

#include <vital/algo/write_track_descriptor_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CPPDB_EXPORT write_track_descriptor_set_db
  : public kwiver::vital::algo::write_track_descriptor_set
{
public:
  // Registered as "db" in register_algorithms.cxx
  PLUGGABLE_IMPL(
    write_track_descriptor_set_db,
    "Writes track descriptors to a database.",
    PARAM_DEFAULT(
      conn_str, std::string,
      "Database connection string.",
      "" ),
    PARAM_DEFAULT(
      video_name, std::string,
      "Name of the video the descriptors belong to.",
      "" ),
    PARAM_DEFAULT(
      write_world_loc, bool,
      "Write world coordinates alongside image coordinates.",
      false ),
    PARAM_DEFAULT(
      commit_interval, unsigned int,
      "Number of writes to batch into a single transaction.",
      1 ) )
  virtual ~write_track_descriptor_set_db();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();
  virtual void write_set( const kwiver::vital::track_descriptor_set_sptr set,
                          const std::string& source_id );

private:
  void initialize() override;
  void set_configuration_internal(
    kwiver::vital::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_CPPDB_WRITE_TRACK_DESCRIPTOR_SET_DB_H
