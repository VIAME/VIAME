/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Object tracker process registration
 *
 * Imported from kwiver in P5-T04: the five track processes from
 * `sprokit/processes/core`. The processes themselves are unchanged and are
 * still in kwiver's namespace; what moved is where they are built and where
 * they register.
 */

#include "viame_processes_object_trackers_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "convert_tracks_to_detections_process.h"
#include "initialize_object_tracks_process.h"
#include "merge_track_sets_process.h"
#include "track_objects_process.h"
#include "unwrap_detections_process.h"

extern "C"
VIAME_PROCESSES_OBJECT_TRACKERS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_object_trackers" );

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
    kwiver::track_objects_process, "track_objects",
    "Tracks detected objects across frames." )

  VIAME_REGISTER_PROCESS(
    kwiver::initialize_object_tracks_process, "initialize_object_tracks",
    "Initialize new object tracks given detections for the current frame." )

  VIAME_REGISTER_PROCESS(
    kwiver::merge_track_sets_process, "merge_track_sets",
    "Merge multiple input track sets into one output set." )

  VIAME_REGISTER_PROCESS(
    kwiver::convert_tracks_to_detections_process, "convert_tracks_to_detections",
    "Convert input object track sets into detection sets for each frame." )

  VIAME_REGISTER_PROCESS(
    kwiver::unwrap_detections_process, "unwrap_detections",
    "Unwrap object detections from object tracks." )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
