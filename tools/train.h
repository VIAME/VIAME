/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_TRAIN_H
#define VIAME_TOOLS_TRAIN_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT train_applet
  : public kwiver::tools::kwiver_applet
{
public:
  train_applet();

  PLUGIN_INFO( "train",
               "Train detector or tracker models.\n\n"
               "This tool trains one of several object detectors or trackers in the system." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_TRAIN_H
