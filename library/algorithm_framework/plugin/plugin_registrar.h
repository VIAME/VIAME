// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PLUGIN_REGISTRY_PLUGIN_REGISTRAR_H
#define PLUGIN_REGISTRY_PLUGIN_REGISTRAR_H

#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>

#if !defined( KWIVER_DEFAULT_PLUGIN_ORGANIZATION )
#define KWIVER_DEFAULT_PLUGIN_ORGANIZATION "undefined"
#endif

// ----------------------------------------------------------------------------
// Support for adding factories to the registry

namespace viame {

class registry;

/// Class to assist in registering tools.
class plugin_registrar
{
public:
  /// Create registrar
  ///
  /// This class contains the common data used for registering tools.
  ///
  /// \param vpl Reference to the registry
  /// \param name Name of this loadable module.
  plugin_registrar(
    viame::registry& vpl,
    const std::string& name )
    : mod_name( name ),
      mod_organization( KWIVER_DEFAULT_PLUGIN_ORGANIZATION ),
      m_registry( vpl )
  {}

  virtual ~plugin_registrar() = default;

  /// Check if module is loaded.
  virtual bool
  is_module_loaded()
  {
    return m_registry.is_module_loaded( mod_name );
  }

  /// Mark module as loaded.
  virtual void
  mark_module_as_loaded()
  {
    m_registry.mark_module_as_loaded( mod_name );
  }

  /// Return module name.
  const std::string&
  module_name() const { return this->mod_name; }

  /// Return module owning organization.
  const std::string&
  organization() const { return this->mod_organization; }

  /// Return reference to the registry.
  viame::registry&
  registry()
  {
    return this->m_registry;
  }

private:
  const std::string mod_name;
  const std::string mod_organization;

  viame::registry& m_registry;
};

} // namespace viame

#endif // PLUGIN_REGISTRY_PLUGIN_REGISTRAR_H
