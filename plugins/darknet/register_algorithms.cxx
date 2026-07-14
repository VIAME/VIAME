/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_darknet_plugin_export.h"

#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/image_object_detector.h>
#include <vital/algo/train_detector.h>

#include "darknet_detector.h"
#include "darknet_trainer.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_DARKNET_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.darknet";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_object_detector, darknet_detector >(
    darknet_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::train_detector, darknet_trainer >(
    darknet_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace
