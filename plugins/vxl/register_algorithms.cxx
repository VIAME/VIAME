/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief VXL plugin algorithm registration interface impl
 */

#include "viame_vxl_plugin_export.h"
#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/image_filter.h>

#include "enhance_images.h"
#include "perform_white_balancing.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_VXL_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.vxl";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::image_filter, enhance_images >(
    enhance_images::_plugin_name );
  fact->add_attribute( kvpf::PLUGIN_NAME, enhance_images::_plugin_name )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  fact = vpm.add_factory< kv::algo::image_filter, perform_white_balancing >(
    perform_white_balancing::_plugin_name );
  fact->add_attribute( kvpf::PLUGIN_NAME, perform_white_balancing::_plugin_name )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
