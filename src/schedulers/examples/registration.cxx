/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "sync_scheduler.h"
#include "thread_per_process_scheduler.h"
#include "thread_pool_scheduler.h"

#include <vistk/pipeline/scheduler_registry.h>

/**
 * \file examples/registration.cxx
 *
 * \brief Register schedulers for use.
 */

using namespace vistk;

void
register_schedulers()
{
  static scheduler_registry::module_t const module_name = scheduler_registry::module_t("example_schedulers");

  scheduler_registry_t const registry = scheduler_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_scheduler("sync", "Run the pipeline synchronously", create_scheduler<sync_scheduler>);
  registry->register_scheduler("thread_per_process", "Run each process in its own thread", create_scheduler<thread_per_process_scheduler>);
  registry->register_scheduler("thread_pool", "Use a pool of threads to step processes", create_scheduler<thread_pool_scheduler>);

  registry->mark_module_as_loaded(module_name);
}
