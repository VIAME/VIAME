/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_PROCESSING_CONVERT_COLOR_SPACE_H
#define VIAME_IMAGE_PROCESSING_CONVERT_COLOR_SPACE_H

#include "viame_image_processing_export.h"

#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

namespace viame {

/**
 * @brief Convert between color spaces in opencv.
 */
class VIAME_IMAGE_PROCESSING_EXPORT convert_color_space
  : public viame::algo::image_filter
{
public:
  PLUGGABLE_IMPL_NAMED(
    convert_color_space, "ocv_convert_color",
                  "Convert image between color spaces",
    PARAM_DEFAULT( input_color_space, std::string,
                   "Input color space.", "RGB" ),
    PARAM_DEFAULT( output_color_space, std::string,
                   "Output color space.", "HLS" )
  )

  virtual ~convert_color_space() = default;

  virtual bool check_configuration( viame::config_block_sptr config ) const;

  // Main filtering method
  virtual viame::image_container_sptr filter(
    viame::image_container_sptr image_data );

protected:
  /// Resolve the conversion code from the defaults at construction
  void initialize() override;

  /// Re-resolve it whenever the configuration changes
  void set_configuration_internal(
    viame::config_block_sptr config ) override;

private:
  void resolve_conversion_code();

  int m_conversion_code = -1;
};

} // end namespace

#endif /* VIAME_IMAGE_PROCESSING_CONVERT_COLOR_SPACE_H */
