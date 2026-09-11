/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_object_detectors_darknet_plugin_export.h"

#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include <viame/algorithm_framework/algo/image_object_detector.h>

#include "darknet_detector.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OBJECT_DETECTORS_DARKNET_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.object_detectors.darknet";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_object_detector,
                               darknet_detector >(
    darknet_detector::plugin_name() );

  fact->add_attribute( kvpf::PLUGIN_NAME, darknet_detector::plugin_name() )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Detect objects with a darknet YOLO network" );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
