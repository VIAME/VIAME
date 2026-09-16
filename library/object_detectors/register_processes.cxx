/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Object detector process registration
 *
 * Imported from `sprokit/processes/core` in P5-T04. The processes are
 * unchanged and are still in kwiver's namespace; what moved is where they
 * are built and where they register.
 *
 * `detect_in_subregions` came from the `opencv` plugin in P2-T05, where it had
 * stopped needing OpenCV in P7-T04b.
 */

#include "viame_processes_object_detectors_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "detect_in_subregions_process.h"
#include "detect_motion_process.h"
#include "image_object_detector_process.h"

extern "C"
VIAME_PROCESSES_OBJECT_DETECTORS_EXPORT
void
register_factories( viame::registry& vpm )
{
  static auto const module_name =
    viame::plugin_manager::module_t( "viame_processes_object_detectors" );

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
    viame::image_object_detector_process, "image_object_detector",
    "Apply selected image object detector algorithm to incoming images." )

  VIAME_REGISTER_PROCESS(
    viame::detect_motion_process, "detect_motion",
    "Detect motion in a sequence of images." )

  VIAME_REGISTER_PROCESS(
    viame::detect_in_subregions_process, "detect_in_subregions",
    "Run a detection algorithm on all of the chips represented "
    "by an incoming detected_object_set" )

#undef VIAME_REGISTER_PROCESS

  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
