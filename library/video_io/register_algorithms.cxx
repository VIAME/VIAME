/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief In-house video and image reader registration
 */

#include "viame_video_io_plugin_export.h"

#include <viame/algorithm_framework/algo/image_io.h>

#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "core_image_io.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_VIDEO_IO_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.video_io";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#define VIAME_REGISTER( interface, impl, name )                      \
  {                                                                  \
    auto fact = vpm.add_factory< interface, impl >( name );          \
    fact->add_attribute( kvpf::PLUGIN_NAME, name )                   \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )        \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,                      \
                      impl::plugin_description() );                  \
  }

  VIAME_REGISTER( kv::algo::image_io, core_image_io,
                  core_image_io::plugin_name() )
  // The name arrows/vxl used for its image reader, kept working now that it
  // is gone. Aliasing it to the plain ocv reader, as lite-removals.md section
  // 1 first suggested, would have dropped all five of its config keys
  VIAME_REGISTER( kv::algo::image_io, core_image_io, "vxl" )

  // The video reader and writer, and the ffmpeg and vidl_ffmpeg names they
  // answer to, are python: see pyav_video_input.py and pyav_video_output.py

#undef VIAME_REGISTER

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
