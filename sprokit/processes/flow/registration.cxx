// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/flow/processes_flow_export.h>

#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "collate_process.h"
#include "distribute_process.h"
#include "pass_process.h"
#include "sink_process.h"
#include "mux_process.h"

/**
 * \file flow/registration.cxx
 *
 * \brief Register processes for use.
 */

extern "C"
PROCESSES_FLOW_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  sprokit::process_registrar reg( vpm, "flow_processes" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< sprokit::collate_process>();
  reg.register_process< sprokit::distribute_process>();
  reg.register_process< sprokit::pass_process>();
  reg.register_process< sprokit::sink_process>();
  reg.register_process< sprokit::mux_process>();

  // - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
