/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Generic pipeline process registration
 *
 * The processes that belong to the pipeline framework itself rather than to
 * any one kind of algorithm. `downsample` came across with sprokit in
 * P5-T03 but was still registered by kwiver's `kwiver_processes` plugin
 * until P5-T04 emptied `sprokit/processes/core`.
 */

#include "viame_processes_pipeline_framework_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "downsample_process.h"
#include "filter_frame_process.h"
#include "filter_frame_index_process.h"
#include "image_to_image_set_process.h"

extern "C"
VIAME_PROCESSES_PIPELINE_FRAMEWORK_EXPORT
void
register_factories( viame::registry& vpm )
{
  static auto const module_name =
    viame::plugin_manager::module_t(
      "viame_processes_pipeline_framework" );

  if( viame::pipeline::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  using kvpf = viame::plugin_factory;

// The parameters are spelled unusually because `typeid( x ).name()` is in
// the body: a parameter called `name` would be substituted inside it.
#define VIAME_REGISTER_PROCESS( process_type, plugin, blurb )             \
  {                                                                      \
    auto* fact = new viame::pipeline::cpp_process_factory(                       \
      typeid( process_type ).name(),                                     \
      viame::pipeline::process::interface_name(),                                \
      viame::pipeline::create_new_process< process_type > );                     \
                                                                         \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb )                  \
      .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );                     \
                                                                         \
    vpm.add_factory( fact );                                             \
  }

  VIAME_REGISTER_PROCESS(
    viame::downsample_process, "downsample",
    "Downsample an input stream." )

  VIAME_REGISTER_PROCESS(
    viame::core::filter_frame_process, "filter_frames",
    "Filter frames based on some property" )

  VIAME_REGISTER_PROCESS(
    viame::core::filter_frame_index_process, "filter_frame_index",
    "Pass frame in min max index limits" )

  VIAME_REGISTER_PROCESS(
    viame::core::image_to_image_set_process, "image_to_image_set",
    "Convert single image to image_set" )

#undef VIAME_REGISTER_PROCESS

  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
