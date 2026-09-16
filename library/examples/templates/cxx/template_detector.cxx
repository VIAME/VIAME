/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "@template@_detector.h"

#include <viame/core_types/detected_object_set.h>

namespace viame {

// -----------------------------------------------------------------------------
bool
@template@_detector
::check_configuration( viame::config_block_sptr config ) const
{
  //++ check for conditions that would prevent the detector from running
  //++ correctly, not necessarily limited to configuration problems
  if( get_text().empty() )
  {
    LOG_ERROR( logger(), "text must not be empty" );
    return false;
  }

  return true;
}


// -----------------------------------------------------------------------------
viame::detected_object_set_sptr
@template@_detector
::detect( viame::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< viame::detected_object_set >();

  //++ insert detector code here
  LOG_INFO( logger(), "Text: " << get_text() );

  return detected_set;
}

} // end namespace viame
