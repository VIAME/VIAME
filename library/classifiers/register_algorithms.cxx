/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Classifier and detection filter registration
 *
 * VIAME's own refiners and mergers came from the `core` plugin in P2-T05. They
 * are declared with PLUGGABLE_IMPL and name and describe themselves, so they
 * register through the template rather than the macro.
 */

#include "viame_classifiers_plugin_export.h"

#include <viame/algorithm_framework/algo/detected_object_filter.h>
#include <viame/algorithm_framework/algo/merge_detections.h>
#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/algorithm_framework/algo/refine_tracks.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "class_probability_filter.h"
#include "convert_head_tail_points.h"
#include "merge_detections_suppress_in_regions.h"
#include "refine_detections_add_fixed.h"
#include "refine_detections_nms.h"
#include "refine_tracks_average_tot.h"
#include "windowed_refiner.h"


namespace viame {

namespace kv = viame;

extern "C"
VIAME_CLASSIFIERS_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.classifiers";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Imported from arrows/core in P5-T04, under the name it registered under
  // there.
#define VIAME_REGISTER_IMPORTED( interface, impl, plugin, blurb )        \
  {                                                                      \
    auto fact = vpm.add_factory< interface, impl >( plugin );            \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );                 \
  }

  VIAME_REGISTER_IMPORTED(
    kv::algo::detected_object_filter,
    viame::core::class_probability_filter,
    "class_probability_filter",
    "Filter detections by the probability of their classes" )

#undef VIAME_REGISTER_IMPORTED

  register_algorithm< kv::algo::refine_detections,
    convert_head_tail_points >( vpm, module_name );
  register_algorithm< kv::algo::merge_detections,
    merge_detections_suppress_in_regions >( vpm, module_name );
  register_algorithm< kv::algo::refine_detections,
    refine_detections_add_fixed >( vpm, module_name );
  register_algorithm< kv::algo::refine_detections,
    refine_detections_nms >( vpm, module_name );
  register_algorithm< kv::algo::refine_tracks,
    refine_tracks_average_tot >( vpm, module_name );

  // One chipper, two names; see `object_detectors/register_algorithms.cxx`.
  register_algorithm< kv::algo::refine_detections,
    windowed_refiner >( vpm, module_name );
  register_alias< kv::algo::refine_detections,
    windowed_refiner >( vpm, module_name, "ocv_windowed" );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
