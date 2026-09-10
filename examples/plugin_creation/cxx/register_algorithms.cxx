/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

/**
 * \file
 * \brief Register algorithms
 */

#include <viame/algorithm_framework/plugin/plugin_loader.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>

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
  using kvpf = kwiver::vital::plugin_factory;

  static auto const module_name = std::string( "viame.example_external_detector" );

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // The interface the implementation is registered against, then the
  // implementation, then the name a pipeline selects it by.
  auto fact = vpm.add_factory< kwiver::vital::algo::image_object_detector,
                               viame::example_detector >( "example_detector" );

  fact->add_attribute( kvpf::PLUGIN_NAME, "example_detector" )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    "Example externally created plugin." )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_VERSION, "1.0" )
    .add_attribute( kvpf::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  // - - - - - - -
  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
