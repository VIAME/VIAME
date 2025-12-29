// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <plugins/burnout/viame_burnout_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "burnout_detector.h"
#include "burnout_image_enhancer.h"
#include "burnout_pixel_classification.h"
#include "burnout_track_descriptors.h"

namespace viame {

extern "C"
VIAME_BURNOUT_PLUGIN_EXPORT
void
register_factories( ::kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "viame.burnout" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< ::viame::burnout_detector >();
  reg.register_algorithm< ::viame::burnout_track_descriptors >();
  reg.register_algorithm< ::viame::burnout_image_enhancer >();
  reg.register_algorithm< ::viame::burnout_pixel_classification >();

  reg.mark_module_as_loaded();
}

} // end namespace viame
