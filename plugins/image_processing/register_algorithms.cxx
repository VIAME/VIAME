/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief In-house image processing algorithm registration
 */

#include "viame_image_processing_plugin_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/plugin_loader.h>

#include "average_frames.h"
#include "color_commonality.h"
#include "convert_image.h"
#include "morphology.h"
#include "threshold.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_IMAGE_PROCESSING_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.image_processing";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#define VIAME_REGISTER_IMAGE_FILTER( impl )                              \
  {                                                                      \
    auto fact = vpm.add_factory< kv::algo::image_filter, impl >(          \
      impl::plugin_name() );                                             \
    fact->add_attribute( kvpf::PLUGIN_NAME, impl::plugin_name() )        \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,                          \
                      impl::plugin_description() );                      \
  }

  VIAME_REGISTER_IMAGE_FILTER( average_frames )
  VIAME_REGISTER_IMAGE_FILTER( color_commonality )
  VIAME_REGISTER_IMAGE_FILTER( convert_image )
  VIAME_REGISTER_IMAGE_FILTER( morphology )
  VIAME_REGISTER_IMAGE_FILTER( threshold )

#undef VIAME_REGISTER_IMAGE_FILTER

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
