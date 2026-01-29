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
  static auto const module_name = kwiver::vital::plugin_manager::module_t( "viame_processes_svm" );
  kwiver::vital::plugin_factory_handle_t fact_handle;
    if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::svm::process_query_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::svm::process_query_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "svm_process_query" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );
  vpm.add_factory( fact );

  fact = new sprokit::cpp_process_factory(
    typeid( viame::svm::train_svm_models_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::svm::train_svm_models_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "svm_train" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
