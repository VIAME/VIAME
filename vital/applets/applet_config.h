// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOLS_APPLET_CONFIG_H
#define KWIVER_TOOLS_APPLET_CONFIG_H

#include <kwiversys/SystemTools.hxx>

#include <vital/applets/kwiver_applet.h>
#include <vital/config/config_block.h>

namespace kwiver {
namespace tools {

/// Load and merge the appropriate default video configuration based on filename
inline kwiver::vital::config_block_sptr
load_default_video_input_config(std::string const& video_file_name)
{
  typedef kwiversys::SystemTools ST;
  typedef kwiver::tools::kwiver_applet kvt;
  if (ST::GetFilenameLastExtension(video_file_name) == ".txt")
  {
    return kvt::find_configuration("core_image_list_video_input.conf");
  }
  return kvt::find_configuration("ffmpeg_video_input.conf");
}

} } // end namespace

#endif
