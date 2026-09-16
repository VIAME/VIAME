/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "viame_processes_seagis_export.h"
#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "seagis_measurement_process.h"

// -----------------------------------------------------------------------------
/*! \brief Registers SEAGIS processes
 *
 */
extern "C"
VIAME_PROCESSES_SEAGIS_EXPORT
void
register_factories( viame::registry& vpm )
{
  using namespace viame::pipeline;
  static auto const module_name = viame::plugin_manager::module_t( "viame_processes_seagis" );
  viame::plugin_factory_handle_t fact_handle;
    if( viame::pipeline::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  // ---------------------------------------------------------------------------
  using kvpf = viame::plugin_factory;

  viame::plugin_factory* fact = new viame::pipeline::cpp_process_factory(
    typeid( viame::seagis::seagis_measurement_process ).name(),
    viame::pipeline::process::interface_name(),
    viame::pipeline::create_new_process< viame::seagis::seagis_measurement_process > );
  fact->add_attribute( kvpf::PLUGIN_NAME, "seagis_measurement" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Read measurements from SEAGIS annotation files" )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );
  vpm.add_factory( fact );

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
