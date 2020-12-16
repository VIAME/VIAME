// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Matlab algorithm registration implementation
 */

#include <arrows/matlab/kwiver_algo_matlab_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/matlab/matlab_image_object_detector.h>
#include <arrows/matlab/matlab_image_filter.h>
#include <arrows/matlab/matlab_detection_output.h>

namespace kwiver {
namespace arrows {
namespace matlab {

extern "C"
KWIVER_ALGO_MATLAB_PLUGIN_EXPORT
void
register_factories( ::kwiver::vital::plugin_loader& vpm )
{
  ::kwiver::vital::algorithm_registrar reg( vpm, "arrows.matlab" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< ::kwiver::arrows::matlab::matlab_image_object_detector >();
  reg.register_algorithm< ::kwiver::arrows::matlab::matlab_image_filter >();
  reg.register_algorithm< ::kwiver::arrows::matlab::matlab_detection_output >();

  reg.mark_module_as_loaded();
}

} } } // end namespace
