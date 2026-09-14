/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_core_plugin_export.h"
#include <viame/algorithm_framework/plugin/registry.h>

#include "adaptive_tracker_trainer.h"
#include "adaptive_detector_trainer.h"
#include "average_track_descriptors.h"
#include "convert_head_tail_points.h"
#include "empty_detector.h"
#include "full_frame_detector.h"
#include "merge_detections_suppress_in_regions.h"
#include "equalize_via_percentiles.h"
#include "query_track_descriptor_set_csv.h"
#include "refine_detections_add_fixed.h"
#include "refine_detections_nms.h"
#include "refine_tracks_average_tot.h"
#include "windowed_detector.h"
#include "windowed_refiner.h"
#include "windowed_trainer.h"

namespace viame {

namespace kv = kwiver::vital;

namespace {

static auto const module_name         = std::string{ "viame.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// Register algorithm using PLUGGABLE_IMPL (plugin_name()/plugin_description())
template <typename interface_t, typename algorithm_t>
void register_algorithm( kv::registry& vpm )
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
register_factories( kv::registry& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Algorithms using PLUGGABLE_IMPL
  register_algorithm< kv::algo::refine_detections,
    convert_head_tail_points >( vpm );
  register_algorithm< kv::algo::image_object_detector,
    empty_detector >( vpm );
  register_algorithm< kv::algo::query_track_descriptor_set,
    query_track_descriptor_set_csv >( vpm );

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
  register_algorithm< kv::algo::refine_tracks,
    refine_tracks_average_tot >( vpm );
  register_algorithm< kv::algo::image_object_detector,
    windowed_detector >( vpm );
  register_algorithm< kv::algo::refine_detections,
    windowed_refiner >( vpm );
  register_algorithm< kv::algo::train_detector,
    windowed_trainer >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
