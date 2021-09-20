/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

/**
 * \file
 * \brief Register algorithms
 */

#include <vital/algo/algorithm_factory.h>

#include "example_detector.h"

namespace viame {

#ifdef WIN32
#define PLUGIN_EXPORT_FLAG __declspec( dllexport )
#else
#define PLUGIN_EXPORT_FLAG __attribute__((visibility("default")))
#endif

extern "C"
PLUGIN_EXPORT_FLAG
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "viame.example_external_detector" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory              implementation-name                 type-to-create
  auto fact = vpm.ADD_ALGORITHM( "example_detector",         viame::example_detector );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Example externally created plugin.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  // - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
