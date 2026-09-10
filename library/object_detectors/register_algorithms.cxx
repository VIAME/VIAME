/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Object detector registration
 */

#include "viame_object_detectors_plugin_export.h"

#include <viame/algorithm_framework/algo/detect_motion.h>
#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "hough_circle_detector.h"
#include "detect_heat_map.h"
#include "detect_motion_3frame_differencing.h"


namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OBJECT_DETECTORS_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.object_detectors";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#define VIAME_REGISTER( interface, impl, plugin, blurb )             \
  {                                                                  \
    auto fact = vpm.add_factory< interface, impl >( plugin );        \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                 \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )        \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );             \
  }

  VIAME_REGISTER( kv::algo::image_object_detector,
                  kwiver::arrows::ocv::hough_circle_detector,
                  "hough_circle", "Detect circles with OpenCV's Hough transform" )

  VIAME_REGISTER( kv::algo::image_object_detector,
                  kwiver::arrows::ocv::detect_heat_map,
                  "detect_heat_map", "Detect regions of a heat map above a threshold" )

  VIAME_REGISTER( kv::algo::detect_motion,
                  kwiver::arrows::ocv::detect_motion_3frame_differencing,
                  "ocv_3frame_differencing", "Detect motion by differencing three frames with OpenCV" )


#undef VIAME_REGISTER

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
