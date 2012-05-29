/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "component_score_json_writer_process.h"
#include "mask_scoring_process.h"
#include "score_aggregation_process.h"
#include "score_writer_process.h"

#include <vistk/pipeline/process_registry.h>

/**
 * \file scoring/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("scoring_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("component_score_json_writer", "Write component-ized scores in JSON.", create_process<component_score_json_writer_process>);
  registry->register_process("mask_scoring", "A process which scores masks.", create_process<mask_scoring_process>);
  registry->register_process("score_aggregation", "A process which aggregates scores.", create_process<score_aggregation_process>);
  registry->register_process("score_writer", "A process which writes out scores.", create_process<score_writer_process>);

  registry->mark_module_as_loaded(module_name);
}
