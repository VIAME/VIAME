/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_habcam
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_HABCAM_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_HABCAM_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_habcam
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with write_detected_object_set_viame_csv
  PLUGGABLE_IMPL(
    read_detected_object_set_habcam,
    "Reads habcam-style detection/ground truth files.",
    PARAM_DEFAULT(
      delimiter, std::string,
      "Delimiter character used in the input file. If empty, auto-detect.",
      "" ),
    PARAM_DEFAULT(
      point_dilation, double,
      "Dilation in pixels applied to point annotations to create bounding boxes.",
      50.0 ),
    PARAM_DEFAULT(
      use_internal_table, bool,
      "Use internal species code lookup table for label mapping.",
      false )
  )

  read_detected_object_set_habcam();
  virtual ~read_detected_object_set_habcam();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_HABCAM_H
