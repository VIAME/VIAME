/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_CONVERT_COLOR_SPACE_H
#define VIAME_OPENCV_CONVERT_COLOR_SPACE_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

/**
 * @brief Convert between color spaces in opencv.
 */
class VIAME_OPENCV_EXPORT convert_color_space
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL( convert_color_space,
                  image_filter,
                  "ocv_convert_color",
                  "Convert image between color spaces",
    PARAM_DEFAULT( input_color_space, std::string,
                   "Input color space.", "RGB" )
    PARAM_DEFAULT( output_color_space, std::string,
                   "Output color space.", "HLS" )
  )

  virtual ~convert_color_space() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  int m_conversion_code = -1;
};

} // end namespace

#endif /* VIAME_OPENCV_CONVERT_COLOR_SPACE_H */
