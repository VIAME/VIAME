/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_core_plugin_export.h"
#include <vital/algo/algorithm_factory.h>

#include "add_timestamp_from_filename.h"
#include "auto_detect_transform.h"
#include "average_track_descriptors.h"
#include "convert_head_tail_points.h"
#include "empty_detector.h"
#include "full_frame_detector.h"
#include "merge_detections_suppress_in_regions.h"
#include "read_detected_object_set_fishnet.h"
#include "read_detected_object_set_habcam.h"
#include "read_detected_object_set_oceaneyes.h"
#include "read_detected_object_set_viame_csv.h"
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

namespace {

static auto const module_name         = std::string{ "viame.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// ----------------------------------------------------------------------------
template <typename algorithm_t>
void register_algorithm( kwiver::vital::plugin_loader& vpm )
{
  using kvpf = kwiver::vital::plugin_factory;

  auto fact = vpm.ADD_ALGORITHM( algorithm_t::name, algorithm_t );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::description )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

extern "C"
VIAME_CORE_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< add_timestamp_from_filename >( vpm );
  register_algorithm< auto_detect_transform_io >( vpm );
  register_algorithm< convert_head_tail_points >( vpm );
  register_algorithm< empty_detector >( vpm );
  register_algorithm< read_detected_object_set_fishnet >( vpm );
  register_algorithm< read_detected_object_set_habcam >( vpm );
  register_algorithm< read_detected_object_set_oceaneyes >( vpm );
  register_algorithm< read_detected_object_set_viame_csv >( vpm );
  register_algorithm< read_object_track_set_viame_csv >( vpm );
  register_algorithm< write_detected_object_set_viame_csv >( vpm );
  register_algorithm< write_disparity_maps >( vpm );
  register_algorithm< write_object_track_set_viame_csv >( vpm );

  // Algorithms using PLUGIN_INFO macro
  ::kwiver::vital::algorithm_registrar reg( vpm, module_name );
  reg.register_algorithm< ::viame::average_track_descriptors >();
  reg.register_algorithm< ::viame::full_frame_detector >();
  reg.register_algorithm< ::viame::merge_detections_suppress_in_regions >();
  reg.register_algorithm< ::viame::refine_detections_add_fixed >();
  reg.register_algorithm< ::viame::refine_detections_nms >();
  reg.register_algorithm< ::viame::windowed_detector >();
  reg.register_algorithm< ::viame::windowed_refiner >();
  reg.register_algorithm< ::viame::windowed_trainer >();

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
