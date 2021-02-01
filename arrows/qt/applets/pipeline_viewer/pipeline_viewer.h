// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_PIPELINE_VIEWER_H
#define KWIVER_TOOL_PIPELINE_VIEWER_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace kwiver {
namespace tools {

class pipeline_viewer : public kwiver_applet
{
public:
  PLUGIN_INFO( "pipe-gui",
               "Run pipelines in a simple GUI.\n\n"
               "This program provides a simple Qt-based front-end "
               "for executing pipelines and viewing images produced by the same." );

  int run() override;
};

} // namespace tools
} // namespace kwiver

#endif
