// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_TOOL_DUMP_KLV_H
#define KWIVER_TOOL_DUMP_KLV_H

#include <vital/applets/kwiver_applet.h>

#include <string>
#include <vector>

namespace kwiver {
namespace arrows {
namespace core {

class dump_klv
  : public kwiver::tools::kwiver_applet
{
public:
  dump_klv();

  PLUGIN_INFO("dump-klv",
              "Dump KLV stream from video.\n\n"
              "This program displays the KLV metadata packets that are embedded in "
              "a video stream.");

  int run() override;
  void add_command_options() override;

}; // end of class

} } } // end namespace

#endif /* KWIVER_TOOL_DUMP_KLV_H */
