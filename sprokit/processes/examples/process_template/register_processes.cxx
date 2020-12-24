// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/process_factory.h>

// -- list processes to register --
#include "template_process.h"
//++ list additional processes here

// ----------------------------------------------------------------
/** \brief Regsiter processes
 *
 *
 */
extern "C"   //++ This needs to have 'C' linkage so the loader can find it.
TEMPLATE_PROCESSES_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  // The process registrar does all the hard work of registering the
  // process with the plugin loader.
  sprokit::process_registrar reg( vpm, "template_process" );

  // Check to see if module is already loaded. If so, then don't do again.
  if ( reg.is_module_loaded() )
  {
    return;
  }

  // ----------------------------------------------------------------
  // The process registrar registers the specified process type.
  reg.register_process< group_ns::template_process >();

  //++ Add more additional processes here.

// - - - - - - - - - - - - - - - - - - - - - - -
  reg.mark_module_as_loaded();
}
