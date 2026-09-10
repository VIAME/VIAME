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
 */

#include "viame_processes_image_processing_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "draw_detected_object_set_process.h"
#include "image_filter_process.h"
#include "merge_images_process.h"
#include "split_image_process.h"
#include "stabilize_image_process.h"

extern "C"
VIAME_PROCESSES_IMAGE_PROCESSING_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
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

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
