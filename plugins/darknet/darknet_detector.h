/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_DARKNET_DETECTOR_H
#define VIAME_DARKNET_DETECTOR_H

#include "viame_darknet_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

class VIAME_DARKNET_EXPORT darknet_detector :
  public kwiver::vital::algo::image_object_detector
{
public:
#define VIAME_DARKNET_DD_PARAMS \
    PARAM_DEFAULT( \
      net_config, std::string, \
      "Name of network config file.", \
      "" ), \
    PARAM_DEFAULT( \
      weight_file, std::string, \
      "Name of optional weight file.", \
      "" ), \
    PARAM_DEFAULT( \
      class_names, std::string, \
      "Name of file that contains the class names.", \
      "" ), \
    PARAM_DEFAULT( \
      thresh, float, \
      "Threshold value.", \
      0.24f ), \
    PARAM_DEFAULT( \
      hier_thresh, float, \
      "Hier threshold value.", \
      0.5f ), \
    PARAM_DEFAULT( \
      gpu_index, int, \
      "GPU index. Only used when darknet is compiled with GPU support.", \
      -1 ), \
    PARAM_DEFAULT( \
      resize_option, std::string, \
      "Pre-processing resize option, can be: disabled, maintain_ar, scale, " \
      "chip, chip_and_original, or adaptive.", \
      "disabled" ), \
    PARAM_DEFAULT( \
      scale, double, \
      "Image scaling factor used when resize_option is scale or chip.", \
      1.0 ), \
    PARAM_DEFAULT( \
      chip_step, int, \
      "When in chip mode, the chip step size between chips.", \
      100 ), \
    PARAM_DEFAULT( \
      nms_threshold, double, \
      "Non-maximum suppression threshold.", \
      0.4 ), \
    PARAM_DEFAULT( \
      gs_to_rgb, bool, \
      "Convert input greyscale images to rgb before processing.", \
      true ), \
    PARAM_DEFAULT( \
      chip_edge_filter, int, \
      "If using chipping, filter out detections this pixel count near borders.", \
      0 ), \
    PARAM_DEFAULT( \
      chip_adaptive_thresh, int, \
      "If using adaptive selection, total pixel count at which we start to chip.", \
      2000000 )

  PLUGGABLE_VARIABLES( VIAME_DARKNET_DD_PARAMS )
  PLUGGABLE_CONSTRUCTOR( darknet_detector, VIAME_DARKNET_DD_PARAMS )

  static std::string plugin_name() { return "darknet"; }
  static std::string plugin_description() { return "Image object detector using darknet."; }

  PLUGGABLE_STATIC_FROM_CONFIG( darknet_detector, VIAME_DARKNET_DD_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_DARKNET_DD_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( darknet_detector, VIAME_DARKNET_DD_PARAMS )

  virtual ~darknet_detector();

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  void initialize() override;
  void set_configuration_internal( kwiver::vital::config_block_sptr config ) override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace

#endif /* VIAME_DARKNET_DETECTOR_H */
