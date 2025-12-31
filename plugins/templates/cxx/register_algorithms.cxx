/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Default plugin algorithm registration interface impl
 */

#include "viame_@template_lib@_plugin_export.h"
#include <vital/algo/algorithm_factory.h>

#include "@template@_detector.h"

namespace viame {

extern "C"
VIAME_@TEMPLATE_LIB@_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "viame.@template_lib@" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "@template@_detector", viame::@template@ );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "template detector")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Your organization" )
    ;


  // - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
