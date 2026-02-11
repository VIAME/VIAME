/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Default plugin algorithm registration interface impl
 */

#include "viame_@template_lib@_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/image_object_detector.h>

#include "@template@_detector.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_@TEMPLATE_LIB@_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.@template_lib@";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_object_detector, @template@_detector >(
    "@template@_detector" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
