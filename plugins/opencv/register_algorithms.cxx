/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_opencv_plugin_export.h"
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include <viame/algorithm_framework/algo/compute_stereo_depth_map.h>
#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/algo/optimize_cameras.h>
#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/algorithm_framework/algo/split_image.h>
#include <viame/algorithm_framework/algo/train_detector.h>
#include <viame/algorithm_framework/algo/warp_image.h>

#include "add_keypoints_from_mask.h"
#include "apply_color_correction.h"
#include "classify_fish_hierarchical_svm.h"
#include "convert_color_space.h"
#include "debayer_filter.h"
#include "enhance_images.h"
#include "random_hue_shift.h"
#include "split_image_habcam.h"
#include "split_image_horizontally.h"
#include "windowed_detector.h"
#include "windowed_refiner.h"
#include "warp_image_ocv.h"
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

  // `ocv_stereo_disparity` is `library/measurement/ocv_stereo_disparity.py`
  // since P7-T06.

  fact = vpm.add_factory< kv::algo::image_filter, convert_color_space >(
    convert_color_space::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, debayer_filter >(
    debayer_filter::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // `ocv_grabcut` and `ocv_watershed` are
  // `library/image_processing/ocv_segmenters.py` since P7-T04b.

  // `ocv_detect_calibration_targets` is
  // `library/measurement/ocv_calibration_targets.py` since P7-T06.

  fact = vpm.add_factory< kv::algo::image_filter, enhance_images >(
    enhance_images::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // plugins/vxl carried a second class also called viame::enhance_images,
  // registered as vxl_enhancer. The two had the same mangled symbols, so the
  // loader bound one definition for both factories and vxl_enhancer ran
  // whichever plugin happened to load first. They produced identical output
  // on every recorded case, so the name is kept here as an alias of this one
  // and the duplicate class is gone.
  fact = vpm.add_factory< kv::algo::image_filter, enhance_images >(
    "vxl_enhancer" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // `ocv_optimize_stereo_cameras` is
  // `library/measurement/ocv_optimize_stereo_cameras.py` since P7-T06, with
  // `filter_stereo_feature_tracks` and `kmedians` beside it as
  // `stereo_frame_selection.py`.

  fact = vpm.add_factory< kv::algo::image_filter, random_hue_shift >(
    random_hue_shift::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::split_image, split_image_habcam >(
    split_image_habcam::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::split_image, split_image_horizontally >(
    split_image_horizontally::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_object_detector, ocv_windowed_detector >(
    ocv_windowed_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, ocv_windowed_refiner >(
    ocv_windowed_refiner::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::train_detector, ocv_windowed_trainer >(
    ocv_windowed_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::warp_image, warp_image_ocv >(
    warp_image_ocv::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
