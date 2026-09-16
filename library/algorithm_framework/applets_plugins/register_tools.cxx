// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "config_explorer_export.h"

#include <viame/algorithm_framework/plugin/registry.h>

#include "config_explorer.h"

// ============================================================================
extern "C"
CONFIG_EXPLORER_EXPORT
void
register_factories( viame::registry& vpl )
{
  using namespace viame::tools;
  using kvpf = ::viame::plugin_factory;

  auto fact =
    vpl.add_factory< kwiver_applet, config_explorer >( "explore-config" );
  fact->add_attribute(
    kvpf::PLUGIN_DESCRIPTION,
    "Kwiver vital applets" )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, "vital_tool_group" )
    .add_attribute( kvpf::ALGORITHM_CATEGORY, kvpf::APPLET_CATEGORY );
}
