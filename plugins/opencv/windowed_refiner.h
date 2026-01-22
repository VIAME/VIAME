/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_WINDOWED_REFINER_H
#define VIAME_OPENCV_WINDOWED_REFINER_H


#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * @brief Window an arbitrary detection refiner over an image
 *
 * This process applies a detection refinement algorithm across multiple
 * windowed regions of an image, scaling input detections to each region.
 */
class VIAME_OPENCV_EXPORT windowed_refiner
  : public kwiver::vital::algorithm_impl< windowed_refiner,
      kwiver::vital::algo::refine_detections >
{
public:

  PLUGIN_INFO( "ocv_windowed",
               "Window some other arbitrary refiner across the image" )

  windowed_refiner();
  virtual ~windowed_refiner();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::detected_object_set_sptr refine(
    kwiver::vital::image_container_sptr image_data,
    kwiver::vital::detected_object_set_sptr detections ) const;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace viame

#endif /* VIAME_OPENCV_WINDOWED_REFINER_H */
