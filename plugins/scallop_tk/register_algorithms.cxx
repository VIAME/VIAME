/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_scallop_tk_plugin_export.h"
#include <vital/algo/algorithm_factory.h>

#include "scallop_tk_detector.h"

namespace viame {

extern "C"
VIAME_SCALLOP_TK_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "viame.scallop_tk" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory              implementation-name                 type-to-create
  auto fact = vpm.ADD_ALGORITHM( "scallop_tk",             viame::scallop_tk_detector );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Uses scallop_tk to detect scallops.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  // - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
