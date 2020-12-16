// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/plugin_loader/plugin_manager.h>
#include <righttrack_plugin_export.h>

#include "rt_process_instrumentation.h"

extern "C"
RIGHTTRACK_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "kpi-RightTrack" );

  if ( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  kwiver::vital::plugin_factory_handle_t fact =
    vpm.ADD_FACTORY( sprokit::process_instrumentation, sprokit::rt_process_instrumentation );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "RightTrack")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION
                    "Sprokit process instrumentation implementation using RightTrack.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, "process-instrumentation" )
   ;

  // - - - - - - - - - - - - - - - - - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}
