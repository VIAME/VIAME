/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_DEBAYER_FILTER_H
#define VIAME_OPENCV_DEBAYER_FILTER_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include "viame_algorithm_plugin_interface.h"

namespace viame {

class VIAME_OPENCV_EXPORT debayer_filter
  : public kwiver::vital::algo::image_filter
{
public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( debayer_filter )
  PLUGIN_INFO( "ocv_debayer",
               "OpenCV debayer filter for converting to RGB or grayscale" )
  
  debayer_filter();
  virtual ~debayer_filter();

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

#endif /* VIAME_OPENCV_DEBAYER_FILTER_H */
