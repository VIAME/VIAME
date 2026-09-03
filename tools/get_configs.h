/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_GET_CONFIGS_H
#define VIAME_TOOLS_GET_CONFIGS_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT get_configs_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "get-configs",
               "Extract pipeline and training parameters as JSON.\n\n"
               "Reads KWIVER pipeline (.pipe) or training configuration "
               "(.conf) files, either a single file or a whole directory, and "
               "writes their configuration parameters as JSON." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_GET_CONFIGS_H
