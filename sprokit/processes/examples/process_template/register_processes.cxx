//++ add project specific copyright/license header


#include <sprokit/pipeline/process_registry.h>

// -- list processes to register --
#include "template_process.h"
//++ list additional processes here


extern "C"   //++ This needs to have 'C' linkage so the loader can find it.
TEMPLATE_PROCESSES_EXPORT
void register_processes();


// ----------------------------------------------------------------
/** \brief Regsiter processes
 *
 *
 */
void register_processes()
{
  static sprokit::process_registry::module_t const module_name =
    sprokit::process_registry::module_t( "template_processes" ); //++ <- replace with real name of module

  sprokit::process_registry_t const registry( sprokit::process_registry::self() );

  // Check to see if module is already loaded. If so, then don't do again.
  if ( registry->is_module_loaded( module_name ) )
  {
    return;
  }

  // ----------------------------------------------------------------

  registry->register_process(
    "template",    //++ name of process type as used in pipeline config file. Does *not* contain the word "process"
    "Description of process. Make as long as necessary to fully explain what the process does "
    "and how to use it. Explain specific algorithms used, etc.",
    sprokit::create_process< group_ns::template_process > );

  //++ Add more additional processes here.

// - - - - - - - - - - - - - - - - - - - - - - -
  registry->mark_module_as_loaded( module_name );
}
