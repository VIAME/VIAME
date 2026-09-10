// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for draw_detected_object_set

#ifndef ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H
#define ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/draw_detected_object_set.h>

#include <vector>

namespace kwiver {

namespace arrows {

namespace ocv {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class VIAME_IMAGE_PROCESSING_EXPORT draw_detected_object_set
  : public vital::algo::draw_detected_object_set
{
public:
  PLUGGABLE_IMPL(
    draw_detected_object_set,
    "Draw bounding box around detected objects on supplied image.",

    PARAM_DEFAULT(
      threshold,
      float,
      "min threshold for output (float). "
      "Detections with confidence values below this value are not drawn.",
      -1.0f ),

    PARAM_DEFAULT(
      alpha_blend_prob,
      bool,
      "If true, those who are less likely will be more transparent.",
      true ),

    PARAM_DEFAULT(
      default_line_thickness,
      float,
      "The default line thickness, in pixels.",
      1.0f ),

    PARAM_DEFAULT(
      default_color,
      std::string,
      "The default color for a class (RGB).",
      "0 0 255" ),

    PARAM_DEFAULT(
      custom_class_color,
      std::string,
      "List of class/thickness/color seperated by semicolon. "
      "For example: person/3/255 0 0;car/2/0 255 0. "
      "Color is in RGB.",
      "" ),

    PARAM_DEFAULT(
      select_classes,
      std::string,
      "List of classes to display, separated by a semicolon. For example: person;car;clam",
      "*ALL*" ),

    PARAM_DEFAULT(
      text_scale,
      float,
      "Scaling for the text label. "
      "Font scale factor that is multiplied by the font-specific base size.",
      0.4f ),

    PARAM_DEFAULT(
      text_thickness,
      float,
      "Thickness of the lines used to draw a text.",
      1.0f ),

    PARAM_DEFAULT(
      clip_box_to_image,
      bool,
      "If this option is set to true, the bounding box is clipped to the image bounds.",
      false ),

    PARAM_DEFAULT(
      draw_text,
      bool,
      "If this option is set to true, the class name is drawn next to the detection.",
      true )
  );

  virtual ~draw_detected_object_set();

  bool check_configuration( vital::config_block_sptr config ) const override;

  /// Draw detected object boxes om image.
  ///
  /// @param detected_set Set of detected objects
  /// @param image Boxes are drawn in this image
  ///
  /// @return Image with boxes and other annotations added.
  kwiver::vital::image_container_sptr
  draw(
    kwiver::vital::detected_object_set_sptr detected_set,
    kwiver::vital::image_container_sptr image ) override;

private:
  void set_configuration_internal( vital::config_block_sptr config ) override;
  void initialize() override;
  class priv;

  KWIVER_UNIQUE_PTR( priv, d );
};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr< draw_detected_object_set >
  draw_detected_object_set_sptr;

} // namespace ocv

} // namespace arrows

}     // end namespace

#endif // ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H
