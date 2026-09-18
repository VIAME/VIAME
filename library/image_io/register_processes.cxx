/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Image process registration
 *
 * The two processes that read and write a single image, imported from
 * `sprokit/processes/core` in P5-T04 and split out of `video_io` with the
 * rest of the image side. They are unchanged and still in kwiver's
 * namespace; what moved is where they are built and where they register.
 */

#include "viame_processes_image_io_export.h"

#include <viame/pipeline_framework/process_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "image_file_reader_process.h"
#include "image_writer_process.h"

extern "C"
VIAME_PROCESSES_IMAGE_IO_EXPORT
void
register_factories( viame::registry& vpm )
{
  static auto const module_name =
    viame::plugin_manager::module_t( "viame_processes_image_io" );

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
    viame::image_writer_process, "image_writer",
    "Write image to disk." )

  VIAME_REGISTER_PROCESS(
    viame::image_file_reader_process, "image_file_reader",
    "Reads an image file given the file name." )

#undef VIAME_REGISTER_PROCESS

  viame::pipeline::mark_process_module_as_loaded( vpm, module_name );
}
