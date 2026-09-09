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

  PLUGGABLE_VARIABLES( VIAME_THRESHOLD_PARAMS )
  PLUGGABLE_CONSTRUCTOR( threshold, VIAME_THRESHOLD_PARAMS )

  static std::string plugin_name() { return "threshold"; }
  static std::string
  plugin_description()
  {
    return "Threshold an image into a binary mask";
  }

  PLUGGABLE_STATIC_FROM_CONFIG( threshold, VIAME_THRESHOLD_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_THRESHOLD_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( threshold, VIAME_THRESHOLD_PARAMS )

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
