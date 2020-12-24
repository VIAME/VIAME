// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_PIPELINE_RUNNER_H
#define KWIVER_TOOL_PIPELINE_RUNNER_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace sprokit {
namespace tools {

class pipeline_runner
  : public kwiver::tools::kwiver_applet
{
public:
  pipeline_runner();

  PLUGIN_INFO( "runner",
               "Runs a pipeline");

  int run() override;
  void add_command_options() override;

}; // end of class

} } // end namespace

#endif /* KWIVER_TOOL_PIPELINE_RUNNER_H */
