/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_seagis_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "seagis_measurement_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers SEAGIS processes
 *
 */
extern "C"
VIAME_PROCESSES_SEAGIS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;
    if( sprokit::is_process_module_loaded( vpm, "viame_processes_seagis_export.h" ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::seagis::seagis_measurement_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::seagis::seagis_measurement_process > );
  
  // PLUGIN_NAME will be extracted from process or set manually in add_attribute calls below
  // fact->add_attribute( kvpf::PLUGIN_NAME, "name_here" );  
  
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, "viame_processes_seagis_export.h" );
}
