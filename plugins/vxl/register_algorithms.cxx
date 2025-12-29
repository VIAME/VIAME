/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief VXL plugin algorithm registration interface impl
 */

#include <plugins/vxl/viame_vxl_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "enhance_images.h"
#include "perform_white_balancing.h"

namespace viame {

extern "C"
VIAME_VXL_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "viame.vxl" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_algorithm< enhance_images >();
  reg.register_algorithm< perform_white_balancing >();

  reg.mark_module_as_loaded();
}

} // end namespace viame
