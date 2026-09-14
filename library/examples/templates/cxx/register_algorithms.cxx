/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief @template_lib@ algorithm registration
 */

#include "viame_@template_lib@_plugin_export.h"

#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "@template@_detector.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_@TEMPLATE_LIB@_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  const std::string module_name = "viame.@template_lib@";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Registers under the name and description PLUGGABLE_IMPL declares. A
  // second name for the same implementation is `register_alias`.
  register_algorithm< kv::algo::image_object_detector,
    @template@_detector >( vpm, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
