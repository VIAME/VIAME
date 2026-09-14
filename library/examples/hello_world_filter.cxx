/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "hello_world_filter.h"

namespace viame {

// =================================================================================================

hello_world_filter::
  ~hello_world_filter()
{}


// -------------------------------------------------------------------------------------------------
bool
hello_world_filter::
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
kwiver::vital::image_container_sptr
hello_world_filter::
filter( kwiver::vital::image_container_sptr image_data )
{
  LOG_INFO( logger(), "Text: " << get_text() );

  return image_data;
}


} // end namespace
