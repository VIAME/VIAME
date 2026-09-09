/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_REGISTRY_DUMP_H
#define VIAME_TOOLS_REGISTRY_DUMP_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT registry_dump_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "registry-dump",
               "Dump every registered plugin name as JSON.\n\n"
               "Loads all plugins and writes the algorithms, processes, "
               "clusters, schedulers and applets they register, together "
               "with each one's configuration keys and defaults. The output "
               "is the machine-checkable baseline of what this build "
               "provides." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_REGISTRY_DUMP_H
