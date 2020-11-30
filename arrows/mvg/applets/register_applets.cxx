// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief register mvg applets into a plugin
 */

#include <arrows/mvg/applets/kwiver_algo_mvg_applets_export.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include <arrows/mvg/applets/init_cameras_landmarks.h>
#include <arrows/mvg/applets/track_features.h>

namespace kwiver {
namespace arrows {
namespace mvg {

// ----------------------------------------------------------------------------
extern "C"
KWIVER_ALGO_MVG_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::applet_registrar reg( vpm, "arrows.mvg.applets" );

  if (reg.is_module_loaded())
  {
    return;
  }

  // -- register applets --
  reg.register_tool< init_cameras_landmarks >();
  reg.register_tool< track_features >();

  reg.mark_module_as_loaded();
}

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver
