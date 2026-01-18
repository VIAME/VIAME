/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_fishnet
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_FISHNET_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_FISHNET_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_fishnet
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  // NOTE: Keep description in sync with write_detected_object_set_fishnet
  PLUGGABLE_IMPL(
    read_detected_object_set_fishnet,
    "Detected object set reader using fishnet csv format.\n\n"
    "  - Column(s) 1: Frame ID no image extension ID\n"
    "  - Column(s) 2: Box ID unique to frame\n"
    "  - Column(s) 3-6: Img-bbox(TL_x,TL_y,BR_x,BR_y)"
    "  - Column(s) 7: Label name\n" )

  read_detected_object_set_fishnet();
  virtual ~read_detected_object_set_fishnet();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_FISHNET_H
