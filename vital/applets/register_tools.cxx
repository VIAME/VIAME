// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "vital_applets_export.h"

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include "config_explorer.h"

// ============================================================================
extern "C"
VITAL_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace kwiver::tools;

  kwiver::applet_registrar reg( vpm, "vital_tool_group" );

  if (reg.is_module_loaded())
  {
    return;
  }

  // -- register applets --
  reg.register_tool< config_explorer >();

  reg.mark_module_as_loaded();
}
