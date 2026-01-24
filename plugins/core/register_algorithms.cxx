/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_core_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/compute_track_descriptors.h>
#include <vital/algo/detected_object_set_input.h>
#include <vital/algo/detected_object_set_output.h>
#include <vital/algo/image_io.h>
#include <vital/algo/image_object_detector.h>
#include <vital/algo/merge_detections.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/refine_detections.h>
#include <vital/algo/train_detector.h>
#include <vital/algo/train_tracker.h>
#include <vital/algo/transform_2d_io.h>
#include <vital/algo/write_object_track_set.h>

#include "adaptive_tracker_trainer.h"
#include "adaptive_detector_trainer.h"
#include "add_timestamp_from_filename.h"
#include "auto_detect_transform.h"
#include "average_track_descriptors.h"
#include "convert_head_tail_points.h"
#include "empty_detector.h"
#include "full_frame_detector.h"
#include "merge_detections_suppress_in_regions.h"
#include "read_detected_object_set_auto.h"
#include "read_detected_object_set_cvat.h"
#include "read_detected_object_set_dive.h"
#include "read_detected_object_set_fishnet.h"
#include "read_detected_object_set_habcam.h"
#include "read_detected_object_set_oceaneyes.h"
#include "read_detected_object_set_viame_csv.h"
#include "read_detected_object_set_yolo.h"
#include "read_object_track_set_auto.h"
#include "read_object_track_set_dive.h"
#include "read_object_track_set_viame_csv.h"
#include "refine_detections_add_fixed.h"
#include "refine_detections_nms.h"
#include "windowed_detector.h"
#include "windowed_refiner.h"
#include "windowed_trainer.h"
#include "write_detected_object_set_viame_csv.h"
#include "write_disparity_maps.h"
#include "write_object_track_set_viame_csv.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_CORE_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.core";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_io, add_timestamp_from_filename >(
    add_timestamp_from_filename::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::transform_2d_io, auto_detect_transform_io >(
    auto_detect_transform_io::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, convert_head_tail_points >(
    convert_head_tail_points::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_object_detector, empty_detector >(
    empty_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_auto >(
    read_detected_object_set_auto::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_cvat >(
    read_detected_object_set_cvat::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_dive >(
    read_detected_object_set_dive::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_fishnet >(
    read_detected_object_set_fishnet::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_habcam >(
    read_detected_object_set_habcam::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_oceaneyes >(
    read_detected_object_set_oceaneyes::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_viame_csv >(
    read_detected_object_set_viame_csv::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_yolo >(
    read_detected_object_set_yolo::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // Add alias "viame_csv" for backward compatibility with existing pipeline configs
  fact = vpm.add_factory< kv::algo::detected_object_set_input, read_detected_object_set_viame_csv >(
    "viame_csv" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::read_object_track_set, read_object_track_set_auto >(
    read_object_track_set_auto::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::read_object_track_set, read_object_track_set_dive >(
    read_object_track_set_dive::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::read_object_track_set, read_object_track_set_viame_csv >(
    read_object_track_set_viame_csv::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::detected_object_set_output, write_detected_object_set_viame_csv >(
    write_detected_object_set_viame_csv::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // Add alias "viame_csv" for backward compatibility with existing pipeline configs
  fact = vpm.add_factory< kv::algo::detected_object_set_output, write_detected_object_set_viame_csv >(
    "viame_csv" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_io, write_disparity_maps >(
    write_disparity_maps::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::write_object_track_set, write_object_track_set_viame_csv >(
    write_object_track_set_viame_csv::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // Algorithms using PLUGGABLE_IMPL macro
  fact = vpm.add_factory< kv::algo::train_tracker, adaptive_tracker_trainer >(
    adaptive_tracker_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::train_detector, adaptive_detector_trainer >(
    adaptive_detector_trainer::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::compute_track_descriptors, average_track_descriptors >(
    average_track_descriptors::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_object_detector, full_frame_detector >(
    full_frame_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::merge_detections, merge_detections_suppress_in_regions >(
    merge_detections_suppress_in_regions::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_add_fixed >(
    refine_detections_add_fixed::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_nms >(
    refine_detections_nms::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  // Add alias "nms" for backward compatibility with existing pipeline configs
  fact = vpm.add_factory< kv::algo::refine_detections, refine_detections_nms >(
    "nms" );
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
