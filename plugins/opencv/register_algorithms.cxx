/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_opencv_plugin_export.h"
#include <vital/algo/algorithm_factory.h>

#include "add_keypoints_from_mask.h"
#include "apply_color_correction.h"
#include "compute_stereo_disparity.h"
#include "debayer_filter.h"
#include "detect_calibration_targets.h"
#include "enhance_images.h"
#include "optimize_stereo_cameras.h"
#include "random_hue_shift.h"
#include "refine_detections_grabcut.h"
#include "refine_detections_watershed.h"
#include "split_image_habcam.h"
#include "split_image_horizontally.h"
#include "windowed_detector.h"
#include "windowed_refiner.h"
#include "windowed_trainer.h"
#include "classify_fish_hierarchical_svm.h"

namespace viame {

extern "C"
VIAME_OPENCV_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "viame.opencv" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  reg.register_algorithm< add_keypoints_from_mask >();
  reg.register_algorithm< apply_color_correction >();
  reg.register_algorithm< compute_stereo_disparity >();
  reg.register_algorithm< debayer_filter >();
  reg.register_algorithm< detect_calibration_targets >();
  reg.register_algorithm< enhance_images >();
  reg.register_algorithm< optimize_stereo_cameras >();
  reg.register_algorithm< random_hue_shift >();
  reg.register_algorithm< refine_detections_grabcut >();
  reg.register_algorithm< refine_detections_watershed >();
  reg.register_algorithm< split_image_habcam >();
  reg.register_algorithm< split_image_horizontally >();
  reg.register_algorithm< windowed_detector >();
  reg.register_algorithm< windowed_refiner >();
  reg.register_algorithm< windowed_trainer >();
  reg.register_algorithm< classify_fish_hierarchical_svm >();

  reg.mark_module_as_loaded();
}

} // end namespace viame
