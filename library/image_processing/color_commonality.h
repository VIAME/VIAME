/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_COLOR_COMMONALITY_H
#define VIAME_IMAGE_PROCESSING_COLOR_COMMONALITY_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief How common each pixel's colour is across the image.
///
/// Rare colours come out dark and common ones bright, which is what the
/// motion and proposal pipelines threshold against. Config keys, defaults and
/// results match the `vxl_color_commonality` it replaces, except in grid
/// mode, which that implementation leaves largely unwritten; see filter().
class VIAME_IMAGE_PROCESSING_EXPORT color_commonality
  : public kv::algo::image_filter
{
public:
#define VIAME_COLOR_COMMONALITY_PARAMS \
    PARAM_DEFAULT( \
      color_resolution_per_channel, unsigned, \
      "Resolution of the utilized histogram (per channel) if the input " \
      "contains 3 channels. Must be a power of 2.", \
      8 ), \
    PARAM_DEFAULT( \
      intensity_resolution, unsigned, \
      "Resolution of the utilized histogram if the input image contains " \
      "1 channel. Must be a power of 2.", \
      16 ), \
    PARAM_DEFAULT( \
      output_scale, unsigned, \
      "Scale the output image (which by default is in the range [0,1]) " \
      "by this amount. 0 means the output type maximum.", \
      0 ), \
    PARAM_DEFAULT( \
      grid_image, bool, \
      "Instead of computing the commonality across the whole image, " \
      "compute it independently in a grid of tiles.", \
      false ), \
    PARAM_DEFAULT( \
      grid_resolution_height, unsigned, \
      "Number of tile rows when grid_image is set", \
      5 ), \
    PARAM_DEFAULT( \
      grid_resolution_width, unsigned, \
      "Number of tile columns when grid_image is set", \
      6 )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    color_commonality,
    "color_commonality",
    "Map each pixel to how common its colour is in the image",
    VIAME_COLOR_COMMONALITY_PARAMS )

  virtual ~color_commonality();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::image_container_sptr filter( kv::image_container_sptr image_data ) override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_COLOR_COMMONALITY_H
