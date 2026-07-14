/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_core_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include "adaptive_tracker_trainer.h"
#include "adaptive_detector_trainer.h"
#include "add_timestamp_from_filename.h"
#include "auto_detect_transform.h"
#include "average_track_descriptors.h"
#include "convert_head_tail_points.h"
#include "empty_detector.h"
#include "full_frame_detector.h"
#include "merge_detections_suppress_in_regions.h"
#include "equalize_via_percentiles.h"
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

namespace {

static auto const module_name         = std::string{ "viame.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// Register algorithm using PLUGGABLE_IMPL (plugin_name()/plugin_description())
template <typename interface_t, typename algorithm_t>
void register_algorithm( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;

  auto fact = vpm.add_factory< interface_t, algorithm_t >(
    algorithm_t::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::plugin_description() )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

extern "C"
VIAME_CORE_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Algorithms using PLUGGABLE_IMPL
  register_algorithm< kv::algo::image_io,
    add_timestamp_from_filename >( vpm );
  register_algorithm< kv::algo::transform_2d_io,
    auto_detect_transform_io >( vpm );
  register_algorithm< kv::algo::refine_detections,
    convert_head_tail_points >( vpm );
  register_algorithm< kv::algo::image_object_detector,
    empty_detector >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_auto >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_cvat >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_dive >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_fishnet >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_habcam >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_oceaneyes >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_viame_csv >( vpm );
  register_algorithm< kv::algo::detected_object_set_input,
    read_detected_object_set_yolo >( vpm );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_auto >( vpm );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_dive >( vpm );
  register_algorithm< kv::algo::read_object_track_set,
    read_object_track_set_viame_csv >( vpm );
  register_algorithm< kv::algo::detected_object_set_output,
    write_detected_object_set_viame_csv >( vpm );
  register_algorithm< kv::algo::image_io,
    write_disparity_maps >( vpm );
  register_algorithm< kv::algo::write_object_track_set,
    write_object_track_set_viame_csv >( vpm );

  // Algorithms using PLUGGABLE_IMPL
  register_algorithm< kv::algo::train_tracker,
    adaptive_tracker_trainer >( vpm );
  register_algorithm< kv::algo::train_detector,
    adaptive_detector_trainer >( vpm );
  register_algorithm< kv::algo::compute_track_descriptors,
    average_track_descriptors >( vpm );
  register_algorithm< kv::algo::image_object_detector,
    full_frame_detector >( vpm );
  register_algorithm< kv::algo::merge_detections,
    merge_detections_suppress_in_regions >( vpm );
  register_algorithm< kv::algo::image_filter,
    equalize_via_percentiles >( vpm );
  register_algorithm< kv::algo::refine_detections,
    refine_detections_add_fixed >( vpm );
  register_algorithm< kv::algo::refine_detections,
    refine_detections_nms >( vpm );
  register_algorithm< kv::algo::image_object_detector,
    windowed_detector >( vpm );
  register_algorithm< kv::algo::refine_detections,
    windowed_refiner >( vpm );
  register_algorithm< kv::algo::train_detector,
    windowed_trainer >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
