/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_darknet_plugin_export.h"

#include <vital/algo/algorithm_factory.h>

#include "darknet_detector.h"
#include "darknet_trainer.h"

namespace viame {

extern "C"
VIAME_DARKNET_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "viame.darknet" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_algorithm< darknet_detector >();
  reg.register_algorithm< darknet_trainer >();

  reg.mark_module_as_loaded();
}

} // end namespace
