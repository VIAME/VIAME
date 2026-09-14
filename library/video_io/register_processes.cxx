/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Video and image process registration
 *
 * Imported from kwiver in P5-T04: the four video and image processes from
 * `sprokit/processes/core` and `image_viewer` from `sprokit/processes/ocv`.
 * The processes themselves are unchanged and are still in kwiver's
 * namespace; what moved is where they are built and where they register.
 *
 * `detect_shot_breaks` and `read_habcam_metadata` joined them from
 * `plugins/core` in P2-T04. They are still in `viame::core` -- the
 * namespaces are normalised once at the end of phase 2, not capability by
 * capability.
 */

#include "viame_processes_video_io_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "detect_shot_breaks_process.h"
#include "frame_list_process.h"
#include "image_file_reader_process.h"
#include "image_writer_process.h"
#include "read_habcam_metadata_process.h"
#include "video_input_process.h"
#include "video_output_process.h"

extern "C"
VIAME_PROCESSES_VIDEO_IO_EXPORT
void
register_factories( kwiver::vital::registry& vpm )
{
  static auto const module_name =
    kwiver::vital::plugin_manager::module_t( "viame_processes_video_io" );

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
    kwiver::video_input_process, "video_input",
    "Reads video files and produces sequential images with metadata per "
    "frame." )

  VIAME_REGISTER_PROCESS(
    kwiver::video_output_process, "video_output",
    "Writes video file based on sequential images with optional metadata "
    "per frame." )

  VIAME_REGISTER_PROCESS(
    kwiver::image_writer_process, "image_writer",
    "Write image to disk." )

  VIAME_REGISTER_PROCESS(
    kwiver::image_file_reader_process, "image_file_reader",
    "Reads an image file given the file name." )

  VIAME_REGISTER_PROCESS(
    kwiver::frame_list_process, "frame_list_input",
    "Reads a list of image file names and generates stream of "
    "images and associated time stamps." )

  VIAME_REGISTER_PROCESS(
    viame::core::detect_shot_breaks_process, "detect_shot_breaks",
    "Detect shot breaks and create tracks for each shot" )

  VIAME_REGISTER_PROCESS(
    viame::core::read_habcam_metadata_process, "read_habcam_metadata",
    "Read HabCam metadata from input files" )

#undef VIAME_REGISTER_PROCESS

  sprokit::mark_process_module_as_loaded( vpm, module_name );
}
