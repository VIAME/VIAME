/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_cvat
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_cvat
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  PLUGGABLE_IMPL(
    read_detected_object_set_cvat,
    "Detected object set reader using CVAT XML format.\n\n"
    "CVAT XML format contains:\n"
    "  - <annotations> root element\n"
    "  - <image> elements with id, name, width, height\n"
    "  - <box> elements with label, xtl, ytl, xbr, ybr\n"
    "  - <polygon> elements with label and points\n\n"
    "Coordinates are in absolute pixels.",
    PARAM_DEFAULT(
      default_confidence, double,
      "Default confidence value for detections (CVAT does not store confidence).",
      1.0 ) )

  virtual ~read_detected_object_set_cvat();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  void initialize() override;
  virtual void new_stream();

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H
