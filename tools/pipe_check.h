/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_PIPE_CHECK_H
#define VIAME_TOOLS_PIPE_CHECK_H

#include <viame/algorithm_framework/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT pipe_check_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "pipe-check",
               "Check that pipeline files still bake and resolve.\n\n"
               "Bakes each .pipe file the way the runner would and reports "
               "every process it contains together with the algorithm "
               "implementation each :type key selects, saying whether that "
               "implementation is registered. .conf files are read and their "
               "algorithm selections resolved the same way." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_PIPE_CHECK_H
