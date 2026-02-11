/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_opencv_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/compute_stereo_depth_map.h>
#include <vital/algo/image_filter.h>
#include <vital/algo/image_object_detector.h>
#include <vital/algo/optimize_cameras.h>
#include <vital/algo/refine_detections.h>
#include <vital/algo/split_image.h>
#include <vital/algo/train_detector.h>

#include "add_keypoints_from_mask.h"
#include "apply_color_correction.h"
#include "classify_fish_hierarchical_svm.h"
#include "compute_stereo_disparity.h"
#include "convert_color_space.h"
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

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OPENCV_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.opencv";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::refine_detections, add_keypoints_from_mask >(
    add_keypoints_from_mask::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, apply_color_correction >(
    apply_color_correction::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, classify_fish_hierarchical_svm >(
    classify_fish_hierarchical_svm::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::compute_stereo_depth_map, compute_stereo_disparity >(
    compute_stereo_disparity::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, convert_color_space >(
    convert_color_space::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, debayer_filter >(
    debayer_filter::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_object_detector, detect_calibration_targets >(
    detect_calibration_targets::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, enhance_images >(
    enhance_images::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::optimize_cameras, optimize_stereo_cameras >(
    optimize_stereo_cameras::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, random_hue_shift >(
    random_hue_shift::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_grabcut >(
    refine_detections_grabcut::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_watershed >(
    refine_detections_watershed::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::split_image, split_image_habcam >(
    split_image_habcam::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::split_image, split_image_horizontally >(
    split_image_horizontally::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_object_detector, windowed_detector >(
    windowed_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, windowed_refiner >(
    windowed_refiner::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::train_detector, windowed_trainer >(
    windowed_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
