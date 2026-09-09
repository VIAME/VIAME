/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_THRESHOLD_H
#define VIAME_IMAGE_PROCESSING_THRESHOLD_H

#include "viame_image_processing_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief Turn an image into a binary mask of the pixels above a value.
///
/// Config keys, defaults and results match the `vxl_threshold` it replaces,
/// except on the one path that implementation left uninitialised; see the
/// note in filter().
class VIAME_IMAGE_PROCESSING_EXPORT threshold
  : public kv::algo::image_filter
{
public:
#define VIAME_THRESHOLD_PARAMS \
    PARAM_DEFAULT( \
      threshold, double, \
      "Threshold to apply. In absolute mode this is a pixel value; in " \
      "percentile mode a fraction in [0, 1].", \
      0.95 ), \
    PARAM_DEFAULT( \
      type, std::string, \
      "Threshold type: absolute or percentile", \
      "percentile" )

  // PLUGGABLE_IMPL_NAMED rather than the pieces spelled out: it is
  // the only spelling that also generates get_configuration, without
  // which a partial config from a pipe file throws on the first key
  // the file does not set
  PLUGGABLE_IMPL_NAMED(
    threshold,
    "threshold",
    "Threshold an image into a binary mask",
    VIAME_THRESHOLD_PARAMS )

  virtual ~threshold();

  bool check_configuration( kv::config_block_sptr config ) const override;

  kv::image_container_sptr filter( kv::image_container_sptr image_data ) override;

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif // VIAME_IMAGE_PROCESSING_THRESHOLD_H
