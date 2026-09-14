/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Registers the SVM query process
 *
 * From `svm` in P2-T08, where it registered beside `train_svm_models`,
 * which is `classifiers`' now. Its own plugin because libsvm is optional.
 */

#include "viame_processes_descriptors_svm_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "process_query_process.h"

extern "C"
VIAME_PROCESSES_DESCRIPTORS_SVM_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_descriptors_svm" );

  if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  using kvpf = kwiver::vital::plugin_factory;

  kwiver::vital::plugin_factory* fact = new sprokit::cpp_process_factory(
    typeid( viame::svm::process_query_process ).name(),
    sprokit::process::interface_name(),
    sprokit::create_new_process< viame::svm::process_query_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "process_query" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Process a query using SVM models" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
