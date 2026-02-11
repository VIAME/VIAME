/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_auto
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_AUTO_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_AUTO_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_auto
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  PLUGGABLE_IMPL(
    read_detected_object_set_auto,
    "Automatic format detection for detected object set reading.\n\n"
    "Detects format based on file extension and content:\n"
    "  - .dive.json -> DIVE JSON\n"
    "  - .coco.json -> COCO JSON\n"
    "  - .json -> Auto-detect DIVE vs COCO\n"
    "  - .csv -> VIAME CSV\n"
    "  - .txt -> YOLO (image list)\n"
    "  - .xml -> CVAT XML\n\n"
    "Delegates to the appropriate specialized reader." )

  virtual ~read_detected_object_set_auto();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  void initialize() override;
  virtual void new_stream();

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_AUTO_H
