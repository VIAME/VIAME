/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Measurement process registration
 *
 * Imported from kwiver in P5-T04: `compute_stereo_depth_map_process` from
 * `sprokit/processes/core`. The process itself is unchanged and is still in
 * kwiver's namespace; what moved is where it is built and where it
 * registers.
 */

#include "viame_processes_measurement_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "compute_stereo_depth_map_process.h"
#include "calibrate_cameras_from_tracks_process.h"
#include "measure_objects_process.h"
#include "pair_stereo_detections_process.h"
#include "refine_measurements_process.h"
#include "ocv_measure_objects_process.h"
#include "ocv_pair_stereo_detections_process.h"
#include "pair_stereo_tracks_process.h"

extern "C"
VIAME_PROCESSES_MEASUREMENT_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_measurement" );

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
    kwiver::compute_stereo_depth_map_process, "compute_stereo_depth_map",
    "Compute a stereo depth map given two frames." )

  VIAME_REGISTER_PROCESS(
    viame::core::calibrate_cameras_from_tracks_process, "calibrate_cameras_from_tracks",
    "Calibrate two cameras from two objects track set" )

  VIAME_REGISTER_PROCESS(
    viame::core::measure_objects_process, "compute_measurements",
    "Stereo measurement process" )

  VIAME_REGISTER_PROCESS(
    viame::core::pair_stereo_detections_process, "pair_stereo_detections",
    "Stereo detection pairing process" )

  VIAME_REGISTER_PROCESS(
    viame::core::refine_measurements_process, "refine_measurements",
    "Refine measurements in object detections via multiple methods" )

  VIAME_REGISTER_PROCESS(
    viame::measure_objects_process, "measure_using_stereo",
    "Stereo measurement process that matches detections between left and right cameras and computes fish length measurements using triangulation" )

  VIAME_REGISTER_PROCESS(
    viame::pair_stereo_detections_process, "ocv_pair_stereo_detections",
    "Compute object detections pair from stereo depth map information" )

  VIAME_REGISTER_PROCESS(
    viame::pair_stereo_tracks_process, "ocv_pair_stereo_tracks",
    "Compute object tracks pair from stereo depth map information" )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
