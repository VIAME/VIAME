// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "kwiver_pipeline_viewer_export.h"

#include <vital/applets/applet_registrar.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "pipeline_viewer.h"

// ----------------------------------------------------------------------------
extern "C"
KWIVER_PIPELINE_VIEWER_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace kwiver::tools;

  kwiver::applet_registrar reg( vpm, "QT_tool_group" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_tool< pipeline_viewer >();

  reg.mark_module_as_loaded();
}
