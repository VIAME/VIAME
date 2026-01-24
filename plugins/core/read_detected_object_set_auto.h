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

#include <memory>

namespace viame {

/// \brief Automatically detect and read detected object sets from various formats.
///
/// This reader automatically detects the input format based on file extension
/// and content inspection, then delegates to the appropriate specialized reader.
///
/// Format detection rules:
///   - .dive.json -> DIVE JSON format
///   - .coco.json -> COCO JSON format
///   - .json -> Auto-detect between DIVE and COCO by inspecting content
///   - .csv -> VIAME CSV format
///   - .txt -> YOLO format (image list with per-image label files)
///   - .xml -> CVAT XML format
///
class VIAME_CORE_EXPORT read_detected_object_set_auto
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "auto";

  static constexpr char const* description =
    "Automatic format detection for detected object set reading.\n\n"
    "Detects format based on file extension and content:\n"
    "  - .dive.json -> DIVE JSON\n"
    "  - .coco.json -> COCO JSON\n"
    "  - .json -> Auto-detect DIVE vs COCO\n"
    "  - .csv -> VIAME CSV\n"
    "  - .txt -> YOLO (image list)\n"
    "  - .xml -> CVAT XML\n\n"
    "Delegates to the appropriate specialized reader.";

  read_detected_object_set_auto();
  virtual ~read_detected_object_set_auto();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual void open( std::string const& filename );
  virtual void close();

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_AUTO_H
