/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_auto
 */

#ifndef VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H
#define VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H

#include "viame_core_export.h"

#include <vital/algo/read_object_track_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_object_track_set_auto
  : public kwiver::vital::algo::read_object_track_set
{
public:
  PLUGGABLE_IMPL(
    read_object_track_set_auto,
    "Auto-detecting object track set reader.\n\n"
    "Detects format from file extension and content:\n"
    "  - .dive.json: DIVE JSON format\n"
    "  - .json: Inspects content (looks for tracks/features)\n"
    "  - .csv: VIAME CSV format\n\n"
    "Delegates to appropriate specialized reader." )

  virtual ~read_object_track_set_auto();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_OBJECT_TRACK_SET_AUTO_H
