// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "sprokit_applets_export.h"

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include "pipeline_runner.h"
#include "pipe_to_dot.h"
#include "pipe_config.h"

// ============================================================================
extern "C"
SPROKIT_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit::tools;

  kwiver::applet_registrar reg( vpm, "sprokit_tool_group" );

  if (reg.is_module_loaded())
  {
    return;
  }

  // -- register applets --
  reg.register_tool< pipeline_runner >();
  reg.register_tool< pipe_to_dot >();
  reg.register_tool< pipe_config >();

  reg.mark_module_as_loaded();
}
