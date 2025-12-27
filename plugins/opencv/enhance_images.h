/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_ENHANCE_IMAGES_H
#define VIAME_OPENCV_ENHANCE_IMAGES_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/image_filter.h>

namespace viame {

class VIAME_OPENCV_EXPORT enhance_images
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGIN_INFO( "ocv_enhancer",
               "Simple illumination normalization using Lab space and CLAHE" )

  enhance_images();
  virtual ~enhance_images();

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

#endif /* VIAME_OPENCV_ENHANCE_IMAGES_H */
