/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_svm_export.h"
#include <sprokit/pipeline/process_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#include "process_query_process.h"
#include "train_svm_models_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers processes
 *
 */
extern "C"
VIAME_PROCESSES_SVM_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  using namespace sprokit;
    if( sprokit::is_process_module_loaded( vpm, "viame_processes_svm_export.h" ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::svm::process_query_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::svm::process_query_process > );
  
  // PLUGIN_NAME will be extracted from process or set manually in add_attribute calls below
  // fact->add_attribute( kvpf::PLUGIN_NAME, "name_here" );  
  
  vpm.add_factory( fact );

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::svm::train_svm_models_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::svm::train_svm_models_process > );
  
  // PLUGIN_NAME will be extracted from process or set manually in add_attribute calls below
  // fact->add_attribute( kvpf::PLUGIN_NAME, "name_here" );  
  
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, "viame_processes_svm_export.h" );
}
