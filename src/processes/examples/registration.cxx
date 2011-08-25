/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "const_process.h"
#include "multiplication_process.h"
#include "mutate_process.h"
#include "number_process.h"
#include "orphan_process.h"
#include "print_number_process.h"
#include "print_string_process.h"

#include <vistk/pipeline/process_registry.h>

#include <boost/make_shared.hpp>

using namespace vistk;

static process_t create_const_process(config_t const& config);
static process_t create_multiplication_process(config_t const& config);
static process_t create_mutate_process(config_t const& config);
static process_t create_number_process(config_t const& config);
static process_t create_orphan_process(config_t const& config);
static process_t create_print_number_process(config_t const& config);
static process_t create_print_string_process(config_t const& config);

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("example_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("const", "A process with the const flag", create_const_process);
  registry->register_process("multiplication", "Multiplies numbers", create_multiplication_process);
  registry->register_process("mutate", "A process with a mutable flag", create_mutate_process);
  registry->register_process("numbers", "Outputs numbers within a range", create_number_process);
  registry->register_process("orphan", "A dummy process", create_orphan_process);
  registry->register_process("print_number", "Print numbers to a file", create_print_number_process);
  registry->register_process("print_string", "Print strings to a file", create_print_string_process);

  registry->mark_module_as_loaded(module_name);
}

process_t
create_const_process(config_t const& config)
{
  return boost::make_shared<const_process>(config);
}

process_t
create_multiplication_process(config_t const& config)
{
  return boost::make_shared<multiplication_process>(config);
}

process_t
create_mutate_process(config_t const& config)
{
  return boost::make_shared<mutate_process>(config);
}

process_t
create_number_process(config_t const& config)
{
  return boost::make_shared<number_process>(config);
}

process_t
create_orphan_process(config_t const& config)
{
  return boost::make_shared<orphan_process>(config);
}

process_t
create_print_number_process(config_t const& config)
{
  return boost::make_shared<print_number_process>(config);
}

process_t
create_print_string_process(config_t const& config)
{
  return boost::make_shared<print_string_process>(config);
}
