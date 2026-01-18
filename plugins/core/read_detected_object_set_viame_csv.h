/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_viame_csv
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_viame_csv
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with write_detected_object_set_viame_csv
  PLUGGABLE_IMPL(
    read_detected_object_set_viame_csv,
    "Detected object set reader using viame_csv format.\n\n"
    "  - Column(s) 1: Detection or Track ID\n"
    "  - Column(s) 2: Video or Image Identifier\n"
    "  - Column(s) 3: Unique Frame Identifier\n"
    "  - Column(s) 4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 8: Detection Confidence\n"
    "  - Column(s) 9: Target Length (0 or less if uncomputed)\n"
    "  - Column(s) 10-11+: Repeated Species, Confidence Pairs\n",
    PARAM_DEFAULT(
      confidence_override, double,
      "If set to a positive value, override all detection confidences with this value.",
      -1.0 ),
    PARAM_DEFAULT(
      poly_to_mask, bool,
      "Convert polygon annotations to detection masks.",
      false ),
    PARAM_DEFAULT(
      warning_file, std::string,
      "If set, write warnings about missing images/annotations to this file.",
      "" )
  )

  read_detected_object_set_viame_csv();
  virtual ~read_detected_object_set_viame_csv();

  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  void initialize() override;

  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_VIAME_CSV_H
