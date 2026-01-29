/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "empty_detector.h"

namespace viame
{

// -----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
empty_detector
::detect( kwiver::vital::image_container_sptr image_data ) const
{
  return std::make_shared< kwiver::vital::detected_object_set >();
}

} // end namespace
