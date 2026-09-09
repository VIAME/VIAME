/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_CONVERT_IMAGE_H
#define VIAME_IMAGE_PROCESSING_CONVERT_IMAGE_H

#include "viame_image_processing_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief Convert an image's pixel format, channel count and range.
///
/// The workhorse of nearly every shipped pipeline: it is what turns a
/// camera's 16 bit or float frames into the 8 bit three channel images the
/// detectors expect. Config keys, defaults and per-pixel results match the
/// `vxl_convert_image` implementation it replaces; see
/// `tests/golden/vxl` for what that means precisely.
class VIAME_IMAGE_PROCESSING_EXPORT convert_image
  : public kv::algo::image_filter
{
public:
#define VIAME_CONVERT_IMAGE_PARAMS \
    PARAM_DEFAULT( \
      format, std::string, \
      "Output type format: byte, sbyte, float, double, uint16, uint32, " \
      "etc. 'copy' keeps the input type and 'disable' passes the image " \
      "through untouched.", \
      "byte" ), \
    PARAM_DEFAULT( \
      single_channel, bool, \
      "Convert input (presumably multi-channel) to single channel", \
      false ), \
    PARAM_DEFAULT( \
      scale_factor, double, \
      "Optional input value scaling factor. A factor of 0 or 1 means no " \
      "scaling, just a cast.", \
      0.0 ), \
    PARAM_DEFAULT( \
      random_grayscale, double, \
      "Convert a fraction of the input images to a 3-channel grayscale, " \
      "as a training augmentation. 0 disables it.", \
      0.0 ), \
    PARAM_DEFAULT( \
      percentile_norm, double, \
      "If set, between [0.0,0.5), stretch the range between this " \
      "percentile and its complement across the output range.", \
      -1.0 ), \
    PARAM_DEFAULT( \
      force_three_channel, bool, \
      "Force the output to be a three channel image", \
      false )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    convert_image,
    "convert_image",
    "Convert image pixel format, channel count and range",
    VIAME_CONVERT_IMAGE_PARAMS )

  virtual ~convert_image();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::image_container_sptr filter( kv::image_container_sptr image_data ) override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_CONVERT_IMAGE_H
