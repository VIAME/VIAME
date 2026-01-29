/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_oceaneyes
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_oceaneyes
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with write_detected_object_set_oceaneyes
  PLUGGABLE_IMPL(
    read_detected_object_set_oceaneyes,
    "Detected object set reader using oceaneyes csv format.\n\n"
    "  - filename, drop id, subject id, n, species identification,\n"
    "  - no fish confidence metric, yes fish confidence metric,\n"
    "  - species ID confidence metric, line confidence metric,\n"
    "  - is overmerged?, can see head/tail, head/tail coordinates",
    PARAM_DEFAULT(
      no_fish_string, std::string,
      "String identifier for no fish entries.",
      "no fish" ),
    PARAM_DEFAULT(
      box_expansion, double,
      "Expansion factor applied to bounding boxes.",
      0.30 ),
    PARAM_DEFAULT(
      max_aspect_ratio, double,
      "Maximum aspect ratio for bounding boxes.",
      2.25 )
  )

  read_detected_object_set_oceaneyes();
  virtual ~read_detected_object_set_oceaneyes();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_OCEANEYES_H
