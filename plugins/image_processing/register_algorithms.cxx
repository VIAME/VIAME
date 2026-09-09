/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief In-house image processing algorithm registration
 */

#include "viame_image_processing_plugin_export.h"

#include <vital/algo/image_filter.h>
#include <vital/algo/image_io.h>
#include <vital/plugin_management/plugin_loader.h>

#include "average_frames.h"
#include "color_commonality.h"
#include "core_image_io.h"
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

  // The names arrows/vxl used to register, kept working now that it is gone.
  // Every one is checked against a recording of what the VXL implementation
  // produced; see tests/golden/vxl
#define VIAME_REGISTER_ALIAS( impl, alias )                              \
  {                                                                      \
    auto fact = vpm.add_factory< kv::algo::image_filter, impl >( alias ); \
    fact->add_attribute( kvpf::PLUGIN_NAME, alias )                      \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,                          \
                      impl::plugin_description() );                      \
  }

  VIAME_REGISTER_ALIAS( average_frames, "vxl_average" )
  VIAME_REGISTER_ALIAS( color_commonality, "vxl_color_commonality" )
  VIAME_REGISTER_ALIAS( convert_image, "vxl_convert_image" )
  VIAME_REGISTER_ALIAS( morphology, "vxl_morphology" )
  VIAME_REGISTER_ALIAS( threshold, "vxl_threshold" )

#undef VIAME_REGISTER_ALIAS
#undef VIAME_REGISTER_IMAGE_FILTER

  {
    auto fact = vpm.add_factory< kv::algo::image_io, core_image_io >(
      core_image_io::plugin_name() );
    fact->add_attribute( kvpf::PLUGIN_NAME, core_image_io::plugin_name() )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                      core_image_io::plugin_description() );
  }

  {
    auto fact = vpm.add_factory< kv::algo::image_io, core_image_io >( "vxl" );
    fact->add_attribute( kvpf::PLUGIN_NAME, "vxl" )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                      core_image_io::plugin_description() );
  }

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
