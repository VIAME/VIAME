/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_itk_plugin_export.h"

#include <vital/plugin_management/plugin_loader.h>

#include <vital/algo/transform_2d_io.h>

#include "ITKTransform.h"

namespace viame
{

namespace kv = kwiver::vital;

extern "C"
VIAME_ITK_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.itk";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  auto fact = vpm.add_factory< kv::algo::transform_2d_io, itk::ITKTransformIO >(
    "itk" );
  fact->add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
