/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_RUN_H
#define VIAME_TOOLS_RUN_H

#include <viame/algorithm_framework/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

// ----------------------------------------------------------------------------
/// The "run" applet: batch processing and single pipeline execution.
///
/// A lone pipe file is executed directly through the pipeline runner, with
/// "pipeline stage N:" markers run as a sequence of pipelines. Any other
/// command line is forwarded to the run.py batch driver.
class VIAME_TOOLS_APPLETS_EXPORT run_applet
  : public kwiver::tools::kwiver_applet
{
public:
  PLUGIN_INFO( "run",
               "Process videos or images, or run a single pipeline file.\n\n"
               "viame run <pipeline.pipe> executes one pipeline; any other "
               "form is batch processing handled by run.py. A model "
               "file (.pt, .pth, .ckpt, .weights, .onnx or .zip) may stand "
               "in for the pipeline and is wrapped in the default detector "
               "or frame classifier." );

  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_RUN_H
