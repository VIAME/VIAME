/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Registering an algorithm, and registering a second name for one.
 *
 * Every library's `register_algorithms.cxx` had its own copy of the first
 * function, five of them identical, once P2-T04 and P2-T05 moved VIAME's own
 * algorithms out of the `core` plugin into the libraries they belong to.
 */

#ifndef VIAME_ALGORITHM_FRAMEWORK_REGISTER_ALGORITHM_H
#define VIAME_ALGORITHM_FRAMEWORK_REGISTER_ALGORITHM_H

#include <viame/algorithm_framework/plugin/registry.h>

#include <string>

namespace viame {

/// Register an algorithm declared with PLUGGABLE_IMPL, which names and
/// describes itself.
template < typename interface_t, typename algorithm_t >
void
register_algorithm( viame::registry& vpm,
                    std::string const& module_name )
{
  using kvpf = viame::plugin_factory;

  auto fact = vpm.add_factory< interface_t, algorithm_t >(
    algorithm_t::plugin_name() );
  fact->add_attribute( kvpf::PLUGIN_NAME, algorithm_t::plugin_name() )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    algorithm_t::plugin_description() );
}

/// Register a second name for an algorithm already registered under its own.
///
/// The alias is a factory like any other -- the same class, constructed the
/// same way -- carrying `PLUGIN_ALIAS_OF` so that `registry-dump` can say
/// what it stands for and `compare_registry` can count the old name as still
/// present.
///
/// Two implementations that turn out to do the same thing become one plus an
/// alias rather than staying two: `windowed` and `ocv_windowed` in P2-T05,
/// where the golden recordings of both were byte-identical.
template < typename interface_t, typename algorithm_t >
void
register_alias( viame::registry& vpm,
                std::string const& module_name,
                std::string const& alias )
{
  using kvpf = viame::plugin_factory;

  auto fact = vpm.add_factory< interface_t, algorithm_t >( alias );
  fact->add_attribute( kvpf::PLUGIN_NAME, alias )
    .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kvpf::PLUGIN_ALIAS_OF, algorithm_t::plugin_name() )
    .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                    algorithm_t::plugin_description() );
}

} // end namespace viame

#endif // VIAME_ALGORITHM_FRAMEWORK_REGISTER_ALGORITHM_H
