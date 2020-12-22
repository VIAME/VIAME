// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_CONFIG_EXPLORER_H
#define KWIVER_TOOL_CONFIG_EXPLORER_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace kwiver {
namespace tools {

class config_explorer
  : public kwiver_applet
{
public:
  config_explorer();

  PLUGIN_INFO( "explore-config",
               "Explore configuration loading process.\n\n"
               "This program assists in debugging config loading problems. It loads a "
               "configuration and displays the contents or displays the search path.");

  int run() override;
  void add_command_options() override;

}; // end of class

} } // end namespace

#endif /* KWIVER_TOOL_CONFIG_EXPLORER_H */
