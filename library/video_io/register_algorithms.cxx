/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief In-house video reader registration
 *
 * The image readers and writers that used to register here are `image_io`'s
 * now. What is left is the image-list video source: it answers to
 * `image_list` as a `video_input`, and reads each frame through whichever
 * `image_io` its config names.
 */

#include "viame_video_io_plugin_export.h"

#include <viame/algorithm_framework/algo/video_input.h>

#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "video_input_image_list.h"

namespace viame {

namespace kv = viame;

extern "C"
VIAME_VIDEO_IO_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.video_io";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // The video reader and writer, and the ffmpeg and vidl_ffmpeg names they
  // answer to, are python: see pyav_video_input.py and pyav_video_output.py

  // Imported from kwiver in P5-T04. Registers the way kwiver's arrows did,
  // by class rather than through a macro, because it is still in
  // kwiver's namespace and carries no plugin_name().
  {
    auto fact = vpm.add_factory< kv::algo::video_input,
      viame::core::video_input_image_list >( "image_list" );
    fact->add_attribute( kvpf::PLUGIN_NAME, "image_list" )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                      "Read a video as a list of image files" );
  }

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
