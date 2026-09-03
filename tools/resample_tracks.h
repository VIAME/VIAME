/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_RESAMPLE_TRACKS_H
#define VIAME_TOOLS_RESAMPLE_TRACKS_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT resample_tracks_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "resample-tracks",
               "Resample object tracks from one frame rate to another.\n\n"
               "Frame numbers are rescaled to the output rate; states missing "
               "at the new rate are filled by interpolating between annotated "
               "states. Track extents are never extrapolated." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_RESAMPLE_TRACKS_H
