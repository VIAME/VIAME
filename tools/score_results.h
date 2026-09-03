/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_SCORE_RESULTS_H
#define VIAME_TOOLS_SCORE_RESULTS_H

#include <vital/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

class VIAME_TOOLS_APPLETS_EXPORT score_results_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "score",
               "Score detection and tracking results against groundtruth.\n\n"
               "Computes precision, recall, F1 and AP, the MOT metrics "
               "(MOTA, MOTP, IDF1), HOTA, and KWANT-style metrics, and can "
               "sweep confidence thresholds to estimate a DIVE filter." );

  void add_command_options() override;

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_SCORE_RESULTS_H
