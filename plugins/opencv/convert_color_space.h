/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_CONVERT_COLOR_SPACE_H
#define VIAME_OPENCV_CONVERT_COLOR_SPACE_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include "viame_algorithm_plugin_interface.h"

namespace viame {

/**
 * @brief Convert between color spaces in opencv.
 */
class VIAME_OPENCV_EXPORT convert_color_space
  : public kwiver::vital::algo::image_filter
{
public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( convert_color_space )
  PLUGIN_INFO( "ocv_convert_color",
               "Convert image between color spaces" )

  convert_color_space();
  virtual ~convert_color_space();

  // Get the current configuration (parameters) for this filter
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_OPENCV_CONVERT_COLOR_SPACE_H */
