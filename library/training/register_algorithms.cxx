/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Trainer registration
 *
 * The adaptive detector and tracker trainers, from the `core` plugin in P2-T07,
 * and the windowed trainer, the merge of the `core` plugin's and
 * the `opencv` plugin's: `ocv_windowed` is an alias of `windowed`, as it is for
 * the windowed detector and refiner.
 */

#include "viame_training_plugin_export.h"

#include <viame/algorithm_framework/algo/train_detector.h>
#include <viame/algorithm_framework/algo/train_tracker.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "adaptive_detector_trainer.h"
#include "adaptive_tracker_trainer.h"
#include "windowed_trainer.h"

namespace viame {

namespace kv = viame;

extern "C"
VIAME_TRAINING_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  const std::string module_name = "viame.training";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< kv::algo::train_tracker,
    adaptive_tracker_trainer >( vpm, module_name );
  register_algorithm< kv::algo::train_detector,
    adaptive_detector_trainer >( vpm, module_name );
  register_algorithm< kv::algo::train_detector,
    windowed_trainer >( vpm, module_name );
  register_alias< kv::algo::train_detector,
    windowed_trainer >( vpm, module_name, "ocv_windowed" );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
