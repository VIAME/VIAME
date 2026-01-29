/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "convert_head_tail_points.h"

#include <cmath>
#include <string>

namespace viame
{

// -----------------------------------------------------------------------------
bool
convert_head_tail_points
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
convert_head_tail_points
::refine( kwiver::vital::image_container_sptr image_data,
  kwiver::vital::detected_object_set_sptr input_dets ) const
{
  auto output = std::make_shared< kwiver::vital::detected_object_set >();

  return output;
}

} // end namespace
