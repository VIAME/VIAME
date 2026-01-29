/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_object_track_set_viame_csv
 */

#ifndef VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H
#define VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/read_object_track_set.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_object_track_set_viame_csv
  : public kwiver::vital::algo::read_object_track_set
{
public:
  PLUGGABLE_IMPL(
    read_object_track_set_viame_csv,
    "Object track set viame_csv reader.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n",
    PARAM_DEFAULT(
      delimiter, std::string,
      "The delimiter used in the CSV file.",
      "," ),
    PARAM_DEFAULT(
      batch_load, bool,
      "Load all tracks at once in batch mode.",
      false ),
    PARAM_DEFAULT(
      confidence_override, double,
      "Override confidence value for all detections (-1.0 to disable).",
      -1.0 ),
    PARAM_DEFAULT(
      poly_to_mask, bool,
      "Convert polygon annotations to detection masks.",
      false ),
    PARAM_DEFAULT(
      frame_id_adjustment, int,
      "Adjustment to add to all frame IDs.",
      0 ),
    PARAM_DEFAULT(
      single_state_only, bool,
      "Only output tracks with a single state.",
      false ),
    PARAM_DEFAULT(
      multi_state_only, bool,
      "Only output tracks with multiple states.",
      false )
  )

  virtual ~read_object_track_set_viame_csv();

  void open( std::string const& filename ) override;

  bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  bool read_set( kwiver::vital::object_track_set_sptr& set ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_OBJECT_TRACK_SET_VIAME_CSV_H
