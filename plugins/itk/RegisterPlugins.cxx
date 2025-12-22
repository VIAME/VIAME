/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <plugins/itk/viame_itk_plugin_export.h>

#include <vital/algo/algorithm_factory.h>

#include "ITKTransform.h"

namespace viame
{

extern "C"
VIAME_ITK_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "viame.itk" );

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory              implementation-name                 type-to-create
  auto fact = vpm.ADD_ALGORITHM( "itk",                 viame::itk::ITKTransformIO );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Load an ITK transformation")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
