/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VXL_ENHANCE_IMAGES_H
#define VIAME_VXL_ENHANCE_IMAGES_H

#include "viame_vxl_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// @brief VXL Image Enhancement
///
/// This method contains basic methods for image filtering on top of input
/// images via automatic white balancing, smoothing, and illumination
/// normalization.
class VIAME_VXL_EXPORT enhance_images
  : public kv::algo::image_filter
{
public:
#define VIAME_VXL_EI_PARAMS \
    PARAM_DEFAULT( \
      disabled, bool, \
      "Completely disable this process and pass the input image", \
      false ), \
    PARAM_DEFAULT( \
      smoothing_enabled, bool, \
      "Perform extra internal smoothing on the input", \
      false ), \
    PARAM_DEFAULT( \
      smoothing_std_dev, double, \
      "Std dev for internal gaussian smoothing", \
      0.6 ), \
    PARAM_DEFAULT( \
      smoothing_half_width, unsigned, \
      "Half width for internal gaussian smoothing", \
      2 ), \
    PARAM_DEFAULT( \
      inversion_enabled, bool, \
      "Should we invert the input image?", \
      false ), \
    PARAM_DEFAULT( \
      auto_white_balance, bool, \
      "Whether or not auto-white balancing is enabled", \
      true ), \
    PARAM_DEFAULT( \
      white_scale_factor, double, \
      "A measure of how much to over or under correct white reference points.", \
      0.95 ), \
    PARAM_DEFAULT( \
      black_scale_factor, double, \
      "A measure of how much to over or under correct black reference points.", \
      0.75 ), \
    PARAM_DEFAULT( \
      exp_history_factor, double, \
      "The exponential averaging factor for correction matrices", \
      0.25 ), \
    PARAM_DEFAULT( \
      matrix_resolution, unsigned, \
      "The resolution of the correction matrix", \
      8 ), \
    PARAM_DEFAULT( \
      normalize_brightness, bool, \
      "If enabled, will attempt to stabilize video illumination", \
      true ), \
    PARAM_DEFAULT( \
      sampling_rate, unsigned, \
      "The sampling rate used when approximating the mean scene illumination.", \
      2 ), \
    PARAM_DEFAULT( \
      brightness_history_length, unsigned, \
      "Attempt to stabilize the brightness using data from the last x frames.", \
      10 ), \
    PARAM_DEFAULT( \
      min_percent_brightness, double, \
      "The minimum allowed average brightness for an image.", \
      0.10 ), \
    PARAM_DEFAULT( \
      max_percent_brightness, double, \
      "The maximum allowed average brightness for an image.", \
      0.90 )

  PLUGGABLE_VARIABLES( VIAME_VXL_EI_PARAMS )
  PLUGGABLE_CONSTRUCTOR( enhance_images, VIAME_VXL_EI_PARAMS )

  static std::string plugin_name() { return "vxl_enhancer"; }
  static std::string plugin_description() { return "Image enhancement using VXL (smoothing, white balance, illumination)"; }

  PLUGGABLE_STATIC_FROM_CONFIG( enhance_images, VIAME_VXL_EI_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_VXL_EI_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( enhance_images, VIAME_VXL_EI_PARAMS )

  virtual ~enhance_images();

  /// Check that the algorithm's configuration is valid
  virtual bool check_configuration( kv::config_block_sptr config ) const;

  /// Perform image enhancement
  virtual kv::image_container_sptr filter(
    kv::image_container_sptr image_data );

  void set_configuration_internal( kv::config_block_sptr config ) override;

private:
  void initialize() override;

  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // end namespace viame

#endif /* VIAME_VXL_ENHANCE_IMAGES_H */
