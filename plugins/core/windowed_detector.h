/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_WINDOWED_DETECTOR_H
#define VIAME_CORE_WINDOWED_DETECTOR_H

#include "viame_core_export.h"

#include <vital/algo/image_object_detector.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary other detector over an image
 *
 * This algorithm wraps another detector and runs it over windowed regions
 * of the input image, then combines the results. This is useful for running
 * detectors that work best on smaller image sizes on larger images.
 *
 * This is a pure vital::image implementation with no OpenCV dependency.
 */
class VIAME_CORE_EXPORT windowed_detector
  : public kwiver::vital::algorithm_impl< windowed_detector,
      kwiver::vital::algo::image_object_detector >
{
public:

  PLUGIN_INFO( "core_windowed",
               "Window some other arbitrary detector across the image (no OpenCV)" )

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

#endif /* VIAME_CORE_WINDOWED_DETECTOR_H */
