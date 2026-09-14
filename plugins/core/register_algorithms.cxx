/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include "viame_core_plugin_export.h"
#include <viame/algorithm_framework/plugin/registry.h>

#include "windowed_trainer.h"

namespace viame {

namespace kv = kwiver::vital;

namespace {

static auto const module_name         = std::string{ "viame.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// Register algorithm using PLUGGABLE_IMPL (plugin_name()/plugin_description())
template <typename interface_t, typename algorithm_t>
void register_algorithm( kv::registry& vpm )
{
  using kvpf = kv::plugin_factory;

  auto fact = vpm.add_factory< interface_t, algorithm_t >(
    algorithm_t::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::plugin_description() )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

extern "C"
VIAME_CORE_PLUGIN_EXPORT
void
register_factories( kv::registry& vpm )
{
  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // Algorithms using PLUGGABLE_IMPL

  // Algorithms using PLUGGABLE_IMPL
  register_algorithm< kv::algo::train_detector,
    windowed_trainer >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
