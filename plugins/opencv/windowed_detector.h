/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_WINDOWED_DETECTOR_H
#define VIAME_OPENCV_WINDOWED_DETECTOR_H


#include "viame_opencv_export.h"

#include <vital/algo/image_object_detector.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary other detector over an image
 *
 * This process should be moved to core from ocv when able
 */
class VIAME_OPENCV_EXPORT windowed_detector
  : public kwiver::vital::algorithm_impl< windowed_detector,
      kwiver::vital::algo::image_object_detector >
{
public:

  PLUGIN_INFO( "ocv_windowed",
               "Window some other arbitrary detector across the image" )

  windowed_detector();
  virtual ~windowed_detector();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_DETECTOR_H */
