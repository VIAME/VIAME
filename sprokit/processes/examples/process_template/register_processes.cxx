//++ add project specific copyright/license header


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
  static const auto module_name = kwiver::vital::plugin_manager::module_t( "template_processes" ); //++ <- replace with real name of module

  // Check to see if module is already loaded. If so, then don't do again.
  if ( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------

  auto fact = vpm.ADD_PROCESS( group_ns::template_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "template" ) //+ use your process name
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Description of process. Make as long as necessary to fully explain what the process does "
                    "and how to use it. Explain specific algorithms used, etc." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  //++ Add more additional processes here.

// - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
