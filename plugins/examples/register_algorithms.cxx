/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Examples plugin algorithm registration interface impl
 */

#include "viame_examples_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/image_object_detector.h>
#include <vital/algo/image_filter.h>

#include "hello_world_detector.h"
#include "hello_world_filter.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_EXAMPLES_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.examples";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_object_detector, hello_world_detector >(
    hello_world_detector::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION, hello_world_detector::plugin_description() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, hello_world_filter >(
    hello_world_filter::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION, hello_world_filter::plugin_description() );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
