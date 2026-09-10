/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Classifier process registration
 *
 * Imported from kwiver in P5-T04: the four refinement and filtering
 * processes from `sprokit/processes/core`. The processes themselves are
 * unchanged and are still in kwiver's namespace; what moved is where they
 * are built and where they register.
 */

#include "viame_processes_classifiers_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "detected_object_filter_process.h"
#include "merge_detection_sets_process.h"
#include "refine_detections_process.h"
#include "refine_tracks_process.h"

extern "C"
VIAME_PROCESSES_CLASSIFIERS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_classifiers" );

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
    kwiver::refine_detections_process, "refine_detections",
    "Refines detections for a given frame," )

  VIAME_REGISTER_PROCESS(
    kwiver::refine_tracks_process, "refine_tracks",
    "Refines object tracks for a given frame" )

  VIAME_REGISTER_PROCESS(
    kwiver::detected_object_filter_process, "detected_object_filter",
    "Filters sets of detected objects using the "
    "detected_object_filter algorithm." )

  VIAME_REGISTER_PROCESS(
    kwiver::merge_detection_sets_process, "merge_detection_sets",
    "Merge multiple input detection sets into one output set.\n\n"
    "This process will accept one or more input ports of detected_object_set "
    "type. They will all be added to the output detection set. "
    "The input port names do not matter since they will be connected "
    "upon connection." )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
