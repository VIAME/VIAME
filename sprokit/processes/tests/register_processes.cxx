// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/tests/kwiver_processes_test_export.h>
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "test_proc_seq.h"

// ----------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
KWIVER_PROCESSES_TEST_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;

  process_registrar reg( vpm, "kwiver_processes_test" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< kwiver::test_proc_seq >();

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
