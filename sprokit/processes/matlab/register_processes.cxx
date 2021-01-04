// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/matlab/kwiver_processes_matlab_export.h>
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "matlab_process.h"

// ----------------------------------------------------------------
/*! \brief Regsiter processes
 *
 *
 */
extern "C"
KWIVER_PROCESSES_MATLAB_EXPORT
void register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;

  process_registrar reg( vpm, "kwiver_processes_matlab" );

  if ( is_process_module_loaded( vpm, reg.module_name() ) )
  {
    return;
  }

  reg.register_process< kwiver::matlab_process >();

// - - - - - - - - - - - - - - - - - - - - - - -
  mark_process_module_as_loaded( vpm, reg.module_name() );
}
