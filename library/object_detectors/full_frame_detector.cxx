/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "full_frame_detector.h"

namespace viame {

namespace kv = kwiver::vital;

bool
full_frame_detector
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

kv::detected_object_set_sptr
full_frame_detector
::detect( kv::image_container_sptr image_data ) const
{
  auto output = std::make_shared< kv::detected_object_set >();

  if( image_data->height() > 0 && image_data->width() > 0 )
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
