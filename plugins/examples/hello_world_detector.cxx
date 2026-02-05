/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "hello_world_detector.h"

namespace viame {

// =================================================================================================

hello_world_detector::
  ~hello_world_detector()
{}


// -------------------------------------------------------------------------------------------------
bool
hello_world_detector::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( get_text().empty() )
  {
    LOG_ERROR( logger(), "Text configuration value cannot be empty" );
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
hello_world_detector::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  LOG_INFO( logger(), "Text: " << get_text() );

  return detected_set;
}


} // end namespace
