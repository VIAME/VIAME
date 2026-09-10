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
 */

#include "viame_processes_object_detectors_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "detect_motion_process.h"
#include "image_object_detector_process.h"

extern "C"
VIAME_PROCESSES_OBJECT_DETECTORS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_object_detectors" );

  if( sprokit::is_process_module_loaded( vpm, module_name ) )
  {
    return;
  }

  using kvpf = kwiver::vital::plugin_factory;

// The parameters are spelled unusually because `typeid( x ).name()` is in
// the body: a parameter called `name` would be substituted inside it.
#define VIAME_REGISTER_PROCESS( process_type, plugin, blurb )             \
  {                                                                      \
    auto* fact = new sprokit::cpp_process_factory(                       \
      typeid( process_type ).name(),                                     \
      sprokit::process::interface_name(),                                \
      sprokit::create_new_process< process_type > );                     \
                                                                         \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb )                  \
      .add_attribute( kvpf::PLUGIN_VERSION, "1.0" );                     \
                                                                         \
    vpm.add_factory( fact );                                             \
  }

  VIAME_REGISTER_PROCESS(
    kwiver::image_object_detector_process, "image_object_detector",
    "Apply selected image object detector algorithm to incoming images." )

  VIAME_REGISTER_PROCESS(
    kwiver::detect_motion_process, "detect_motion",
    "Detect motion in a sequence of images." )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
