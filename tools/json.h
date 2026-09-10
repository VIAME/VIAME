/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_JSON_H
#define VIAME_TOOLS_JSON_H

#include <viame/algorithm_framework/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT json_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "json",
               "Perform filtering and analysis actions on DIVE and COCO JSON files.\n\n"
               "This tool mirrors 'viame csv' for JSON annotation files: frame ID "
               "adjustment, type filtering and replacement, track renumbering, "
               "statistics, and structural validation. The format is detected from "
               "the file contents." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_JSON_H
