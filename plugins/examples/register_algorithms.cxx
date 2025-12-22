/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Examples plugin algorithm registration interface impl
 */

#include <plugins/examples/viame_examples_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include "hello_world_detector.h"
#include "hello_world_filter.h"

namespace viame {

namespace {

static auto const module_name         = std::string{ "viame.examples" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// ----------------------------------------------------------------------------
template <typename algorithm_t>
void register_algorithm( kwiver::vital::plugin_loader& vpm )
{
  using kvpf = kwiver::vital::plugin_factory;

  auto fact = vpm.ADD_ALGORITHM( algorithm_t::name, algorithm_t );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::description )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

extern "C"
VIAME_EXAMPLES_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< hello_world_detector >( vpm );
  register_algorithm< hello_world_filter >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
