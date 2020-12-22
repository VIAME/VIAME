// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PLUGIN_LOADER_PLUGIN_REGISTRAR_H
#define PLUGIN_LOADER_PLUGIN_REGISTRAR_H

#include <vital/plugin_loader/plugin_factory.h>
#include <vital/plugin_loader/plugin_loader.h>

#if ! defined( KWIVER_DEFAULT_PLUGIN_ORGANIZATION )
#define KWIVER_DEFAULT_PLUGIN_ORGANIZATION "undefined"
#endif

// ==================================================================
// Support for adding factories for the plugin loader

namespace kwiver {
namespace vital {
  class plugin_loader;
} // end namespace vital

/// Class to assist in registering tools.
class plugin_registrar
{
public:
  /// Create registrar
  /**
   * This class contains the common data used for registering tools.
   *
   * \param vpl Reference to the plugin loader
   * \param name Name of this loadable module.
   */
  plugin_registrar( vital::plugin_loader& vpl,
                    const std::string& name )
    : mod_name( name )
    , mod_organization( KWIVER_DEFAULT_PLUGIN_ORGANIZATION )
    , m_plugin_loader( vpl )
  {
  }

  virtual ~plugin_registrar() = default;

  /// Check if module is loaded.
  virtual bool is_module_loaded() { return m_plugin_loader.is_module_loaded( mod_name ); }

  /// Mark module as loaded.
  virtual void mark_module_as_loaded() { m_plugin_loader.mark_module_as_loaded( mod_name ); }

  /// Return module name.
  const std::string& module_name() const { return this->mod_name; }

  /// Return module owning organization.
  const std::string& organization() const { return this->mod_organization; }

  /// Return reference to the plugin loader.
  kwiver::vital::plugin_loader& plugin_loader() { return this->m_plugin_loader; }

private:
  const std::string mod_name;
  const std::string mod_organization;

  kwiver::vital::plugin_loader& m_plugin_loader;
};

} // end namespace

#endif // PLUGIN_LOADER_PLUGIN_REGISTRAR_H
