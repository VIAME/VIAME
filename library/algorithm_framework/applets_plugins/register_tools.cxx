// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "config_explorer_export.h"

#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "config_explorer.h"

// ============================================================================
extern "C"
CONFIG_EXPLORER_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpl )
{
  using namespace kwiver::tools;
  using kvpf = ::kwiver::vital::plugin_factory;

  auto fact =
    vpl.add_factory< kwiver_applet, config_explorer >( "explore-config" );
  fact->add_attribute(
    kvpf::PLUGIN_DESCRIPTION,
    "Kwiver vital applets" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, "vital_tool_group" )
    .add_attribute( kvpf::ALGORITHM_CATEGORY, kvpf::APPLET_CATEGORY );
}
