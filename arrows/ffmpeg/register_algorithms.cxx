// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Register VXL algorithms implementation
 */

#include <arrows/ffmpeg/kwiver_algo_ffmpeg_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/ffmpeg/ffmpeg_video_input.h>

namespace kwiver {
namespace arrows {
namespace ffmpeg {

extern "C"
KWIVER_ALGO_FFMPEG_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "arrows.ffmpeg" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< ::kwiver::arrows::ffmpeg::ffmpeg_video_input >();

  reg.mark_module_as_loaded();
}

} // end namespace ffmpeg
} // end namespace arrows
} // end namespace kwiver
