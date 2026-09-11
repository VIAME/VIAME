/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_training_plugin_export.h"

#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include <viame/algorithm_framework/algo/train_detector.h>

#include "darknet_trainer.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_TRAINING_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;

  // This library's own module name rather than the `viame.darknet` these
  // carried in `plugins/darknet`. A module name is what the loader
  // deduplicates on and nothing outside it ever names one -- the baseline
  // does not record them -- and the detector now registers under
  // `viame.object_detectors`, so sharing a name across two libraries would
  // mean whichever loaded second registered nothing.
  const std::string module_name = "viame.training";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::train_detector, darknet_trainer >(
    darknet_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace
