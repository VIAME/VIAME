// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_PIPE_CONFIG_H
#define KWIVER_TOOL_PIPE_CONFIG_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace sprokit {
namespace tools {

class pipe_config
  : public kwiver::tools::kwiver_applet
{
public:
  pipe_config();

  int run() override;
  void add_command_options() override;

  PLUGIN_INFO( "pipe-config",
    "Configures a pipeline\n\n"
    "This tool reads a pipeline configuration file, applies the program options "
    "and generates a \"compiled\" config file. "
    "At its most basic, this tool will validate a pipeline "
    "configuration, but it does so much more.  Specific pipeline "
    "configurations can be generated from generic descriptions. "
    "\n\n"
    "Global config sections can ge inserted in the resulting configuration "
    "file with the --setting option, with multiple options allowed on the "
    "command line. For example, --setting master:value=FOO will generate a "
    "config section: "
    "\n\n"
    "config master\n"
    "  :value FOO\n"
    "\n\b"
    "The --config option specifies a file that contains additional "
    "configuration parameters to be merged into the generated "
    "configuration. "
    "\n\n"
    "Use the --include option to add additional directories to search for "
    "included configuration files. "
    "\n\n"
    "The --pipeline option specifies the file that contains the main pipeline specification"
    );

}; // end of class

} } // end namespace

#endif /* KWIVER_TOOL_PIPE_CONFIG_H */
