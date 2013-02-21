/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "const_number_process.h"
#include "const_process.h"
#include "data_dependent_process.h"
#include "feedback_process.h"
#include "flow_dependent_process.h"
#include "multiplication_process.h"
#include "multiplier_cluster.h"
#include "mutate_process.h"
#include "number_process.h"
#include "orphan_cluster.h"
#include "orphan_process.h"
#include "print_number_process.h"
#include "shared_process.h"
#include "tagged_flow_dependent_process.h"
#include "take_number_process.h"
#include "take_string_process.h"

#include <vistk/pipeline/process_registry.h>

/**
 * \file examples/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("example_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("const", "A process with the const flag", create_process<const_process>);
  registry->register_process("const_number", "Outputs a constant number", create_process<const_number_process>);
  registry->register_process("data_dependent", "A process with a data dependent type", create_process<data_dependent_process>);
  registry->register_process("feedback", "A process which feeds data into itself", create_process<feedback_process>);
  registry->register_process("flow_dependent", "A process with a flow dependent type", create_process<flow_dependent_process>);
  registry->register_process("multiplication", "Multiplies numbers", create_process<multiplication_process>);
  registry->register_process("multiplier_cluster", "A constant factor multiplier cluster", create_process<multiplier_cluster>);
  registry->register_process("mutate", "A process with a mutable flag", create_process<mutate_process>);
  registry->register_process("numbers", "Outputs numbers within a range", create_process<number_process>);
  registry->register_process("orphan_cluster", "A dummy cluster", create_process<orphan_cluster>);
  registry->register_process("orphan", "A dummy process", create_process<orphan_process>);
  registry->register_process("print_number", "Print numbers to a file", create_process<print_number_process>);
  registry->register_process("shared", "A process with the shared flag", create_process<shared_process>);
  registry->register_process("tagged_flow_dependent", "A process with a tagged flow dependent types", create_process<tagged_flow_dependent_process>);
  registry->register_process("take_number", "Print numbers to a file", create_process<take_number_process>);
  registry->register_process("take_string", "Print strings to a file", create_process<take_string_process>);

  registry->mark_module_as_loaded(module_name);
}
