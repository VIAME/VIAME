/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "refine_detections_add_fixed.h"

namespace viame {

namespace kv = kwiver::vital;

/// Check that the algorithm's currently configuration is valid
bool
refine_detections_add_fixed
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------------
kv::detected_object_set_sptr
refine_detections_add_fixed
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  kv::detected_object_set_sptr output = detections
    ? detections->clone()
    : std::make_shared< kv::detected_object_set >();

  if( c_add_full_image_detection && image_data &&
      image_data->height() > 0 && image_data->width() > 0 )
  {
    kv::bounding_box_d det_box( 0, 0,
                                image_data->width(),
                                image_data->height() );

    if( !c_detection_type.empty() )
    {
      auto dot = std::make_shared< kv::detected_object_type >();
      dot->set_score( c_detection_type, 1.0 );

      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0, dot ) );
    }
    else
    {
      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0 ) );
    }
  }

  return output;
}

} // end namespace viame
