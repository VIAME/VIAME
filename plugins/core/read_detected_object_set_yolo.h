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
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <memory>

namespace viame {

class VIAME_CORE_EXPORT read_detected_object_set_yolo
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  PLUGGABLE_IMPL(
    read_detected_object_set_yolo,
    "Detected object set reader using YOLO format.\n\n"
    "YOLO format stores one .txt file per image with detections:\n"
    "  class_id x_center y_center width height [confidence]\n\n"
    "Coordinates are normalized [0,1] and converted using image dimensions.\n\n"
    "For each image, the reader looks for the label file:\n"
    "  1. Same directory as image (image.txt)\n"
    "  2. ../labels/subdir/ directory relative to image\n"
    "  3. ../labels/ directory relative to image\n",
    PARAM_DEFAULT(
      classes_file, std::string,
      "Path to file containing class names, one per line. "
      "Line number corresponds to class ID (0-indexed). "
      "If empty, will search for 'labels.txt' in image directory or parent.",
      "" ),
    PARAM_DEFAULT(
      image_width, int,
      "Width of images in pixels. If 0, will auto-detect from first image.",
      0 ),
    PARAM_DEFAULT(
      image_height, int,
      "Height of images in pixels. If 0, will auto-detect from first image.",
      0 ),
    PARAM_DEFAULT(
      default_confidence, double,
      "Default confidence score to use when not specified in label file.",
      1.0 ) )

  virtual ~read_detected_object_set_yolo();

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

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_YOLO_H
