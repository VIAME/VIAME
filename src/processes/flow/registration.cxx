/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "collate_process.h"
#include "distribute_process.h"

#include <vistk/pipeline/process_registry.h>

using namespace vistk;

static process_t create_collate_process(config_t const& config);
static process_t create_distribute_process(config_t const& config);

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("example_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("collate", "A process which collates data from multiple worker processes.", create_collate_process);
  registry->register_process("distribute", "A process which distributes data to multiple worker processes.", create_distribute_process);

  registry->mark_module_as_loaded(module_name);
}

process_t
create_collate_process(config_t const& config)
{
  return process_t(new collate_process(config));
}

process_t
create_distribute_process(config_t const& config)
{
  return process_t(new distribute_process(config));
}
