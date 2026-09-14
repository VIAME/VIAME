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
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "detect_heat_map.h"
#include "empty_detector.h"
#include "detect_motion_3frame_differencing.h"
#include "example_detector.h"
#include "full_frame_detector.h"
#include "windowed_detector.h"



namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OBJECT_DETECTORS_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
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

  // `hough_circle` is `hough_circle_detector.py` since P7-T04: the Hough
  // gradient transform is cv2's, and only the language changed.

  VIAME_REGISTER( kv::algo::image_object_detector,
                  kwiver::arrows::ocv::detect_heat_map,
                  "detect_heat_map", "Detect regions of a heat map above a threshold" )

  VIAME_REGISTER( kv::algo::detect_motion,
                  kwiver::arrows::ocv::detect_motion_3frame_differencing,
                  "ocv_3frame_differencing", "Detect motion by differencing three frames with OpenCV" )

  VIAME_REGISTER( kv::algo::image_object_detector,
                  kwiver::arrows::core::example_detector,
                  "example_detector", "Detect a fixed box, for testing a pipeline" )

#undef VIAME_REGISTER

  // From the `core` plugin in P2-T05. The detector that returns nothing and the
  // one that returns the whole frame, both of which pipelines use as a
  // stand-in for a real detector.
  register_algorithm< kv::algo::image_object_detector,
    empty_detector >( vpm, module_name );
  register_algorithm< kv::algo::image_object_detector,
    full_frame_detector >( vpm, module_name );

  // One chipper, two names. the `core` plugin and the `opencv` plugin each had an
  // implementation of it, and P2-T05 kept this one; `tests/golden/opencv`
  // holds the seven chipping variants of both, byte-identical. 276 config
  // lines select `ocv_windowed` and 17 select `windowed`, so both answer.
  register_algorithm< kv::algo::image_object_detector,
    windowed_detector >( vpm, module_name );
  register_alias< kv::algo::image_object_detector,
    windowed_detector >( vpm, module_name, "ocv_windowed" );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
