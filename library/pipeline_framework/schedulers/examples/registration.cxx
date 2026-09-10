// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file examples/registration.cxx
 *
 * \brief Register schedulers for use.
 */

#include "thread_pool_scheduler.h"

#include <viame/pipeline_framework/scheduler_factory.h>
#include <schedulers/examples/schedulers_examples_export.h>

extern "C"
SCHEDULERS_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  sprokit::scheduler_registrar reg( vpm, "example_schedulers" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_scheduler< sprokit::thread_pool_scheduler >(
    "thread_pool",
    "Use a pool of threads to step processes. This example is not functional.",
    "0.1" );

  reg.mark_module_as_loaded();
}
