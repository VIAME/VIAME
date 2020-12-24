// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/processes/ocv/kwiver_processes_ocv_export.h>
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

// -- list processes to register --
#include "image_viewer_process.h"

// ----------------------------------------------------------------
/*! \brief Regsiter processes
 *
 */
extern "C"
KWIVER_PROCESSES_OCV_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;

  process_registrar reg( vpm, "kwiver_processes_ocv" );

  if ( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_process< kwiver::image_viewer_process >();

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
