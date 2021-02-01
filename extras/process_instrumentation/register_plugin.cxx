// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <instrumentation_plugin_export.h>

#include "logger_process_instrumentation.h"
#include "timing_process_instrumentation.h"

extern "C"
INSTRUMENTATION_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "kwiver.process_instrumentation" );

  if ( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  kwiver::vital::plugin_factory_handle_t
    fact = vpm.ADD_FACTORY( sprokit::process_instrumentation, sprokit::logger_process_instrumentation );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "logger")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Sprokit process instrumentation implementation using logger.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, "process-instrumentation" )
    ;

  fact = vpm.ADD_FACTORY( sprokit::process_instrumentation, sprokit::timing_process_instrumentation );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "timing")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Sprokit process instrumentation implementation using timer.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, "process-instrumentation" )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}
