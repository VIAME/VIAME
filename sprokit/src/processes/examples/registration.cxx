// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file examples/registration.cxx
 *
 * \brief Register processes for use.
 */

#include <sprokit/pipeline/process_factory.h>

#include "any_source_process.h"
#include "const_process.h"
#include "const_number_process.h"
#include "data_dependent_process.h"
#include "duplicate_process.h"
#include "expect_process.h"
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
#include "skip_process.h"
#include "tagged_flow_dependent_process.h"
#include "take_number_process.h"
#include "take_string_process.h"
#include "tunable_process.h"

extern "C"
PROCESSES_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;

  process_registrar reg( vpm, "sprokit.example_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< any_source_process >();
  reg.register_process< const_number_process >();
  reg.register_process< const_process >();
  reg.register_process< data_dependent_process >();
  reg.register_process< duplicate_process >();
  reg.register_process< expect_process >();
  reg.register_process< feedback_process >();
  reg.register_process< flow_dependent_process >();
  reg.register_process< multiplication_process >();
  reg.register_process< multiplier_cluster >();
  reg.register_process< mutate_process >();
  reg.register_process< number_process >();
  reg.register_process< orphan_cluster >();
  reg.register_process< orphan_process >();
  reg.register_process< print_number_process >();
  reg.register_process< shared_process >();
  reg.register_process< skip_process >();
  reg.register_process< tagged_flow_dependent_process >();
  reg.register_process< take_number_process >();
  reg.register_process< take_string_process >();
  reg.register_process< tunable_process >();

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
