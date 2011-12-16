/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "sync_schedule.h"
#include "thread_per_process_schedule.h"
#include "thread_pool_schedule.h"

#include <vistk/pipeline/schedule_registry.h>

/**
 * \file examples/registration.cxx
 *
 * \brief Register schedules for use.
 */

using namespace vistk;

void
register_schedules()
{
  static schedule_registry::module_t const module_name = schedule_registry::module_t("example_schedules");

  schedule_registry_t const registry = schedule_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_schedule("sync", "Runs the pipeline synchronously", create_schedule<sync_schedule>);
  registry->register_schedule("thread_per_process", "Runs each process in its own thread", create_schedule<thread_per_process_schedule>);
  registry->register_schedule("thread_pool", "Uses a pool of threads to step processes", create_schedule<thread_pool_schedule>);

  registry->mark_module_as_loaded(module_name);
}
