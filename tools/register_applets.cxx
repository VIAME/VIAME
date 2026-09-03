/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Register VIAME tool applets into a plugin
 */

#include "viame_tools_applets_export.h"

#include <vital/plugin_management/plugin_loader.h>
#include <vital/applets/applet_registrar.h>

#include "applet_attributes.h"
#include "csv.h"
#include "get_configs.h"
#include "resample_tracks.h"
#include "score_results.h"
#include "train.h"

#ifdef VIAME_TOOLS_ENABLE_PYTHON
#include "python_script_applet.h"
#endif

namespace viame {
namespace tools {

#ifdef VIAME_TOOLS_ENABLE_PYTHON

VIAME_PYTHON_SCRIPT_APPLET( process_video_applet, "process-video",
  "process_video.py", "Process new videos" )

#endif

// ----------------------------------------------------------------------------
/// Register an applet that needs no plugins loaded on its behalf.
template < typename applet_t >
static void
register_standalone_tool( kwiver::applet_registrar& reg )
{
  reg.register_tool< applet_t >()->add_attribute( SKIP_PLUGIN_PRELOAD, "true" );
}

// ----------------------------------------------------------------------------
/// Register an applet that forwards its whole command line to a script.
template < typename applet_t >
static void
register_script_tool( kwiver::applet_registrar& reg )
{
  reg.register_tool< applet_t >()
    ->add_attribute( SKIP_PLUGIN_PRELOAD, "true" )
    .add_attribute( FORWARDS_HELP, "true" );
}

// ----------------------------------------------------------------------------
extern "C"
VIAME_TOOLS_APPLETS_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::applet_registrar reg( vpm, "viame.tools.applets" );

  if( reg.is_module_loaded() )
  {
    return;
  }

  // -- register applets --
  register_standalone_tool< csv_applet >( reg );
  register_standalone_tool< get_configs_applet >( reg );
  register_standalone_tool< resample_tracks_applet >( reg );
  register_standalone_tool< score_results_applet >( reg );
  register_standalone_tool< train_applet >( reg );

#ifdef VIAME_TOOLS_ENABLE_PYTHON
  register_script_tool< process_video_applet >( reg );
#endif

  reg.mark_module_as_loaded();
}

} // namespace tools
} // namespace viame
