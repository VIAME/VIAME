/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "collate_process.h"
#include "distribute_process.h"
#include "sink_process.h"

#include <vistk/pipeline/process_registry.h>

/**
 * \file flow/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("flow_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("collate", "Collates data from multiple worker processes", create_process<collate_process>);
  registry->register_process("distribute", "Distributes data to multiple worker processes", create_process<distribute_process>);
  registry->register_process("sink", "Ignores incoming data", create_process<sink_process>);

  registry->mark_module_as_loaded(module_name);
}
