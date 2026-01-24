/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_yolo
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_YOLO_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_YOLO_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/algo/image_io.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_yolo
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "yolo";

  static constexpr char const* description =
    "Detected object set reader using YOLO format.\n\n"
    "YOLO format stores one .txt file per image with detections:\n"
    "  class_id x_center y_center width height [confidence]\n\n"
    "Coordinates are normalized [0,1] and converted using image dimensions.\n\n"
    "For each image, the reader looks for the label file:\n"
    "  1. Same directory as image (image.txt)\n"
    "  2. ../labels/subdir/ directory relative to image\n"
    "  3. ../labels/ directory relative to image\n";

  read_detected_object_set_yolo();
  virtual ~read_detected_object_set_yolo();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_YOLO_H
