// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief register core applets into a plugin
 */

#include <arrows/core/applets/kwiver_algo_core_applets_export.h>
#include <vital/plugin_loader/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include <arrows/core/applets/dump_klv.h>
#include <arrows/core/applets/render_mesh.h>

namespace kwiver {
namespace arrows {
namespace core {

// ----------------------------------------------------------------------------
extern "C"
KWIVER_ALGO_CORE_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::applet_registrar reg( vpm, "arrows.core.applets" );

  if (reg.is_module_loaded())
  {
    return;
  }

  // -- register applets --
  reg.register_tool< dump_klv >();
  reg.register_tool< render_mesh >();

  reg.mark_module_as_loaded();
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
