// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/adapters/viame_processes_adapter_export.h>
#include <viame/pipeline_framework/process_factory.h>

#include <viame/algorithm_framework/plugin/registry.h>

#include "input_adapter_process.h"
#include "output_adapter_process.h"

// ----------------------------------------------------------------
/** \brief Register processes
 *
 *
 */
extern "C"
VIAME_PROCESSES_ADAPTER_EXPORT
void
register_factories( viame::registry& vpm )
{
  using namespace viame::pipeline;

  process_registrar reg( vpm, "kwiver_processes_adapters" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< viame::input_adapter_process >( process_registrar::no_test );
  reg.register_process< viame::output_adapter_process >( process_registrar::no_test );

  // - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
