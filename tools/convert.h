/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_TOOLS_CONVERT_H
#define VIAME_TOOLS_CONVERT_H

#include <viame/algorithm_framework/applets/kwiver_applet.h>

#include "viame_tools_applets_export.h"

namespace viame {
namespace tools {

/// Convert annotation files between formats, or camera calibration files
///
/// Annotation conversions go straight through the registered vital readers
/// and writers, one file or a whole folder at a time, using imagery found
/// next to the annotations when it is there. Anything that is not an
/// annotation file (stereo calibrations, ITK transforms) is handed to the
/// convert.py script.
class VIAME_TOOLS_APPLETS_EXPORT convert_applet
  : public kwiver::tools::kwiver_applet
{
public:
  convert_applet();

  PLUGIN_INFO( "convert",
               "Convert annotation, calibration and registration files between formats.\n\n"
               "Annotation files (VIAME CSV, COCO and DIVE JSON, KW18, HabCam, "
               "CVAT, ...) are converted through the registered readers and "
               "writers, singly or by folder, with imagery found alongside them "
               "used for frame names and timing. Stereo calibrations and ITK "
               "transforms are converted by the calibration converter." );

  void add_command_options() override;
  int run() override;
};

} // namespace tools
} // namespace viame

#endif // VIAME_TOOLS_CONVERT_H
