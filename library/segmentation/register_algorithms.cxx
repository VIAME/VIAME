/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Segmentation registration
 *
 * One C++ implementation so far. `ocv_grabcut` and `ocv_watershed` went to
 * python in P7-T04b and are declared in `__init__.py`; the SAM family is
 * still in the `pytorch` plugin.
 */

#include "viame_segmentation_plugin_export.h"

#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "add_keypoints_from_mask.h"

namespace viame {

namespace kv = viame;

extern "C"
VIAME_SEGMENTATION_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  const std::string module_name = "viame.segmentation";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< kv::algo::refine_detections,
    add_keypoints_from_mask >( vpm, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
