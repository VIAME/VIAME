/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "multiplication_process.h"
#include "number_process.h"

#include <vistk/pipeline/process_registry.h>

using namespace vistk;

static process_t create_multiplication_process(config_t const& config);
static process_t create_number_process(config_t const& config);

void
register_processes()
{
  process_registry_t const registry = process_registry::self();

  registry->register_process("multiplication", "Multiplies numbers", create_multiplication_process);
  registry->register_process("numbers", "Outputs numbers within a range", create_number_process);
}

process_t
create_multiplication_process(config_t const& config)
{
  return process_t(new multiplication_process(config));
}

process_t
create_number_process(config_t const& config)
{
  return process_t(new number_process(config));
}
