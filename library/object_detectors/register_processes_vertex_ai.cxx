/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <viame/algorithm_framework/vital_config.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include <viame/pipeline_framework/process_factory.h>

#include "vertex_ai_detector.h"
#include "vertex_ai_trainer.h"

extern "C"
VIAME_PROCESSES_VERTEX_AI_EXPORT
void register_factories( viame::registry& vpm )
{
  static auto const module_name =
    viame::plugin_manager::module_t( "viame_processes_vertex_ai" );

  if( viame::pipeline::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  auto fact = vpm.ADD_PROCESS(
    viame::vertex_ai::vertex_ai_detector );
  fact->add_attribute(
      viame::plugin_factory::PLUGIN_NAME,
      "vertex_ai_detector" )
    .add_attribute(
      viame::plugin_factory::PLUGIN_MODULE_NAME,
      module_name )
    .add_attribute(
      viame::plugin_factory::PLUGIN_DESCRIPTION,
      "Send inference requests to a deployed Vertex AI endpoint" )
    .add_attribute(
      viame::plugin_factory::PLUGIN_VERSION, "1.0" );

  fact = vpm.ADD_PROCESS(
    viame::vertex_ai::vertex_ai_trainer );
  fact->add_attribute(
      viame::plugin_factory::PLUGIN_NAME,
      "vertex_ai_trainer" )
    .add_attribute(
      viame::plugin_factory::PLUGIN_MODULE_NAME,
      module_name )
    .add_attribute(
      viame::plugin_factory::PLUGIN_DESCRIPTION,
      "Submit a custom training job to Vertex AI" )
    .add_attribute(
      viame::plugin_factory::PLUGIN_VERSION, "1.0" );

  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
