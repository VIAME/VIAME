// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "viame/pipeline_framework/applets/viame_applets_plugin_export.h"

#include <viame/algorithm_framework/plugin/registry.h>
#include <viame/algorithm_framework/applets/applet_registrar.h>

#include "pipeline_runner.h"

// ============================================================================
extern "C"
VIAME_APPLETS_PLUGIN_EXPORT
void
register_factories( viame::registry& vpm )
{
  using namespace viame::pipeline::tools;

  viame::applet_registrar reg( vpm, "sprokit_tool_group" );

  if (reg.is_module_loaded())
  {
    return;
  }

  // -- register applets --
  reg.register_tool< pipeline_runner >();

  reg.mark_module_as_loaded();
}
