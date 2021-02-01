// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/darknet/kwiver_algo_darknet_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/darknet/darknet_detector.h>
#include <arrows/darknet/darknet_trainer.h>

namespace kwiver {
namespace arrows {
namespace darknet {

extern "C"
KWIVER_ALGO_DARKNET_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "arrows.darknet" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< ::kwiver::arrows::darknet::darknet_detector >();
  reg.register_algorithm< ::kwiver::arrows::darknet::darknet_trainer >();

  reg.mark_module_as_loaded();
}

} } } // end namespace
