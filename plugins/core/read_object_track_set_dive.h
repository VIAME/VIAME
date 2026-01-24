/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_dive
 */

#ifndef VIAME_CORE_READ_OBJECT_TRACK_SET_DIVE_H
#define VIAME_CORE_READ_OBJECT_TRACK_SET_DIVE_H

#include "viame_core_export.h"

#include <vital/algo/read_object_track_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_object_track_set_dive
  : public kwiver::vital::algo::read_object_track_set
{
public:
  PLUGGABLE_IMPL(
    read_object_track_set_dive,
    "Object track set reader using DIVE JSON format.\n\n"
    "DIVE JSON natively stores tracks with temporal features.\n"
    "Format contains:\n"
    "  - tracks: object with track data indexed by ID\n"
    "  - Each track: id, begin, end, confidencePairs, features\n"
    "  - Each feature: frame, bounds [x1,y1,x2,y2], geometry\n"
    "  - confidencePairs: [[label, score], ...]\n\n"
    "See: https://kitware.github.io/dive/DataFormats/",
    PARAM_DEFAULT(
      batch_load, bool,
      "Load all tracks at once (true) or stream frame-by-frame (false).",
      true ) )

  virtual ~read_object_track_set_dive();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );

  virtual bool read_set( kwiver::vital::object_track_set_sptr& set );

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_OBJECT_TRACK_SET_DIVE_H
