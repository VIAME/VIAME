// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_PIPE_TO_DOT_H
#define KWIVER_TOOL_PIPE_TO_DOT_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace sprokit {
namespace tools {

class pipe_to_dot
  : public kwiver::tools::kwiver_applet
{
public:
  pipe_to_dot();

  int run() override;
  void add_command_options() override;

  PLUGIN_INFO( "pipe-to-dot",
               "Create DOT output of pipe topology")

}; // end of class

} } // end namespace

#endif /* KWIVER_TOOL_PIPE_TO_DOT_H */
