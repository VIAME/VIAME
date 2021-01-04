// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file examples/registration.cxx
 *
 * \brief Register schedulers for use.
 */

#include "thread_pool_scheduler.h"

#include <sprokit/pipeline/scheduler_factory.h>
#include <schedulers/examples/schedulers_examples_export.h>

extern "C"
SCHEDULERS_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = kwiver::vital::plugin_manager::module_t("example_schedulers");

  if ( sprokit::is_scheduler_module_loaded( vpm, module_name ) )
  {
    return;
  }

  auto fact = vpm.ADD_SCHEDULER( sprokit::thread_pool_scheduler );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "thread_pool" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use a pool of threads to step processes. "
                       "This example is not functional." );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "0.1" );

  sprokit::mark_scheduler_module_as_loaded( vpm, module_name );
}
