/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_CSV_H
#define VIAME_TOOLS_CSV_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT csv_applet
  : public kwiver::tools::kwiver_applet
{
public:
  csv_applet();

  PLUGIN_INFO( "csv",
               "Perform filtering and analysis actions on VIAME CSV files.\n\n"
               "This tool provides various utilities for manipulating and analyzing "
               "VIAME CSV detection/track files including frame ID adjustment, "
               "type filtering, statistics, and more." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_CSV_H
