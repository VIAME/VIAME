/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_opencv_plugin_export.h"
#include <viame/algorithm_framework/plugin/registry.h>

#include <viame/algorithm_framework/algo/train_detector.h>

#include "windowed_trainer.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OPENCV_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.opencv";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Everything else this file registered is `library/` now, and where each
  // one went is in `design/lite-file-map.tsv`: the colour and split filters
  // and `ocv_warp_image` to `image_processing`, `add_keypoints_from_mask` to
  // `segmentation`, `detect_in_subregions` and the windowed detector to
  // `object_detectors`, the windowed refiner to `classifiers` (P2-T05); the
  // python replacements -- the segmenters, the enhancer, the colour
  // correction, the calibration and stereo pieces -- to `image_processing`
  // and `measurement` in phase 7.
  //
  // The trainer is the last of it, and follows when `library/training`
  // exists. `ocv_windowed` is an alias of `windowed` for the detector and
  // the refiner already; the trainer keeps two implementations until then.
  auto fact = vpm.add_factory< kv::algo::train_detector, ocv_windowed_trainer >(
    ocv_windowed_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
