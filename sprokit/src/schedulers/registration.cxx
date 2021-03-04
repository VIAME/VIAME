// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/scheduler_factory.h>

#include "sync_scheduler.h"
#include "thread_per_process_scheduler.h"

#include <schedulers/schedulers_export.h>

/**
 * \file registration.cxx
 *
 * \brief Register schedulers for use.
 */

extern "C"
SCHEDULERS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t("schedulers");

  if ( sprokit::is_scheduler_module_loaded( vpm, module_name ) )
  {
    return;
  }

  auto fact = vpm.ADD_SCHEDULER( sprokit::sync_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "sync" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Run the pipeline synchronously" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_SCHEDULER( sprokit::thread_per_process_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "thread_per_process" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "Run each process in its own thread" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  sprokit::mark_scheduler_module_as_loaded( vpm, module_name );
}
