/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Image filtering and geometry process registration
 *
 * Imported from `sprokit/processes/core` in P5-T04. The processes are
 * unchanged and are still in kwiver's namespace; what moved is where they
 * are built and where they register.
 *
 * The five that came from the `core` plugin in P2-T05 are still in
 * `viame::core` -- the namespaces are normalised once at the end of phase 2,
 * not capability by capability. `accumulate_image_statistics` had no
 * description and no version where it was; registering it through the macro
 * gives it both, which the registry contract allows (it fails on a name that
 * disappears, not on an attribute that appears).
 */

#include "viame_processes_image_processing_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "accumulate_image_statistics_process.h"
#include "align_multimodal_imagery_process.h"
#include "draw_detected_object_set_process.h"
#include "image_filter_process.h"
#include "merge_images_process.h"
#include "split_image_process.h"
#include "stabilize_image_process.h"
#include "stack_frames_process.h"
#include "warp_detections_process.h"
#include "warp_image_process.h"

extern "C"
VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_image_processing" );

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
    kwiver::image_filter_process, "image_filter",
    "Apply an image filter to an image." )

  VIAME_REGISTER_PROCESS(
    kwiver::split_image_process, "split_image",
    "Split an image into two images." )

  VIAME_REGISTER_PROCESS(
    kwiver::merge_images_process, "merge_images",
    "Merge two images into one." )

  VIAME_REGISTER_PROCESS(
    kwiver::stabilize_image_process, "stabilize_image",
    "Generate current-to-reference image homographies." )

  VIAME_REGISTER_PROCESS(
    kwiver::draw_detected_object_set_process, "draw_detected_object_set",
    "Draw detected object set boxes on an image." )

  VIAME_REGISTER_PROCESS(
    viame::core::accumulate_image_statistics_process,
    "accumulate_image_statistics",
    "Accumulate per-frame image statistics over a sequence" )

  VIAME_REGISTER_PROCESS(
    viame::core::align_multimodal_imagery_process, "align_multimodal_imagery",
    "Align multimodal images that may be out of sync" )

  VIAME_REGISTER_PROCESS(
    viame::core::stack_frames_process, "stack_frames",
    "Stack multiple frames on top of each in the same image" )

  VIAME_REGISTER_PROCESS(
    viame::core::warp_detections_process, "warp_detections",
    "Warp detection bounding boxes with a 2D transform loaded from a file "
    "(DIVE registration .json or plain text homography)" )

  VIAME_REGISTER_PROCESS(
    viame::core::warp_image_process, "warp_image",
    "Warp an image with a 2D homography loaded from a file "
    "(DIVE registration .json or plain text homography)" )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
