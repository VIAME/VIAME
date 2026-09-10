/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Object tracker registration
 */

#include "viame_object_trackers_plugin_export.h"

#include <viame/algorithm_framework/algo/associate_detections_to_tracks.h>
#include <viame/algorithm_framework/algo/initialize_object_tracks.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "associate_detections_to_tracks_threshold.h"
#include "initialize_object_tracks_threshold.h"


namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_OBJECT_TRACKERS_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.object_trackers";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Imported from arrows/core in P5-T04, under the names they registered
  // under there.
#define VIAME_REGISTER_IMPORTED( interface, impl, plugin, blurb )        \
  {                                                                      \
    auto fact = vpm.add_factory< interface, impl >( plugin );            \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );                 \
  }

  VIAME_REGISTER_IMPORTED(
    kv::algo::initialize_object_tracks,
    kwiver::arrows::core::initialize_object_tracks_threshold,
    "threshold", "Start a track for every detection above a threshold" )

  VIAME_REGISTER_IMPORTED(
    kv::algo::associate_detections_to_tracks,
    kwiver::arrows::core::associate_detections_to_tracks_threshold,
    "threshold", "Associate detections to tracks by thresholding a cost matrix" )

#undef VIAME_REGISTER_IMPORTED

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
