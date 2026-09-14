/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Example algorithm registration
 *
 * The two hello world algorithms, registered under the names and
 * descriptions their `PLUGGABLE_IMPL` declares. The example processes
 * beside them register in `register_processes.cxx`.
 */

#include "viame_examples_plugin_export.h"

#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "hello_world_detector.h"
#include "hello_world_filter.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_EXAMPLES_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  const std::string module_name = "viame.examples";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< kv::algo::image_object_detector,
    hello_world_detector >( vpm, module_name );
  register_algorithm< kv::algo::image_filter,
    hello_world_filter >( vpm, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
