// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/scheduler_factory.h>

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
register_factories( viame::registry& vpm )
{
  viame::pipeline::scheduler_registrar reg( vpm, "schedulers" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_scheduler< viame::pipeline::thread_per_process_scheduler >(
    "thread_per_process",
    "Run each process in its own thread" );

  reg.mark_module_as_loaded();
}
