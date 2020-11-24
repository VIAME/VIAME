// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/burnout/kwiver_algo_burnout_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/burnout/burnout_image_enhancer.h>
#include <arrows/burnout/burnout_pixel_classification.h>
#include <arrows/burnout/burnout_track_descriptors.h>

namespace kwiver {
namespace arrows {
namespace burnout {

extern "C"
KWIVER_ALGO_BURNOUT_PLUGIN_EXPORT
void
register_factories( ::kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "arrows.burnout" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< ::kwiver::arrows::burnout::burnout_track_descriptors >();
  reg.register_algorithm< ::kwiver::arrows::burnout::burnout_image_enhancer >();
  reg.register_algorithm< ::kwiver::arrows::burnout::burnout_pixel_classification >();

  reg.mark_module_as_loaded();
}

} } } // end namespace
