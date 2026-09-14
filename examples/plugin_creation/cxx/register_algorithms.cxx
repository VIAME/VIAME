/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

/**
 * \file
 * \brief Register algorithms
 */

#include <viame/algorithm_framework/algo/image_object_detector.h>
#include <viame/algorithm_framework/plugin/register_algorithm.h>
#include <viame/algorithm_framework/plugin/registry.h>

#include "example_detector.h"

#ifdef WIN32
#define PLUGIN_EXPORT_FLAG __declspec( dllexport )
#else
#define PLUGIN_EXPORT_FLAG __attribute__((visibility("default")))
#endif

// The entry point an out-of-tree plugin exports. VIAME calls it for every
// library named in `VIAME_PLUGIN_PATH`; everything VIAME ships registers by
// being linked in instead, and has no entry point to find.
//
// It is deliberately not `register_factories`, which is what the built-in
// registration files define: that name belongs to code that is compiled into
// VIAME, and keeping the two apart means a plugin cannot be half-adopted by
// accident.
extern "C"
PLUGIN_EXPORT_FLAG
void
viame_register_plugin( kwiver::vital::registry& vpm )
{
  static auto const module_name = std::string( "viame.example_external_detector" );

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Registers under the name and description PLUGGABLE_IMPL declares. A
  // second name for the same implementation is `viame::register_alias`.
  viame::register_algorithm< kwiver::vital::algo::image_object_detector,
                             viame::external_example_detector >( vpm, module_name );

  vpm.mark_module_as_loaded( module_name );
}
