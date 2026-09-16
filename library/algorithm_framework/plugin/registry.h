// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_REGISTRY_H_
#define KWIVER_VITAL_REGISTRY_H_

#include <viame/algorithm_framework/plugin/vital_vpm_export.h>

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/core_types/vital_types.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace viame {

// base class of factory hierarchy
class plugin_factory;

using plugin_factory_handle_t = std::shared_ptr< plugin_factory >;
using plugin_factory_vector_t = std::vector< plugin_factory_handle_t >;
using plugin_map_t            = std::map< std::string,
  plugin_factory_vector_t >;
using plugin_module_map_t     = std::map< std::string, std::string >;

class registry_impl;

/**
 * @brief The registry: every factory VIAME was built with, by interface.
 *
 * This was the plugin loader, and it found plugins by scanning directories and
 * `dlopen`-ing what it found. P8-T03 links the plugins in and calls their
 * registration functions directly, so what is left is the store they
 * register into -- factories by interface name, and the set of modules that
 * have already registered. P8-T10 renamed it, because a class called a
 * loader that loads nothing sends every reader looking for the load.
 *
 * It stays in `plugin/` rather than moving to `registry/` next to the
 * generated `register_builtins`, which is where `lite-build-system.md` §4
 * draws it. The two cannot share a library: `registry/` links every library
 * that registers, and every library that registers calls `add_factory` on
 * this class. One of the two has to be underneath the other, and it is this
 * one.
 */
class VITAL_VPM_EXPORT registry
{
public:
  registry();

  virtual ~registry();

  // Factory Stuff =============================================================
  /// @brief Get list of factories for interface type.
  ///
  /// This method returns a list of pointer to factory methods that
  /// create objects of the desired interface type.
  ///
  /// @param type_name Type name of the interface required
  ///
  /// @return Vector of factories. (vector may be empty)
  [[nodiscard]]
  plugin_factory_vector_t const& get_factories(
    std::string const& type_name ) const;

  template < typename INTERFACE >
  [[nodiscard]]

  plugin_factory_vector_t const&
  get_factories() const
  {
    return get_factories( get_interface_name< INTERFACE >() );
  }

  /// @brief Add factory to manager.
  ///
  /// This method adds the specified plugin factory to the plugin
  /// manager. This method is usually called from the plugin
  /// registration function in the loadable module to self-register all
  /// plugins in a module.
  ///
  /// Factory instances provide *MUST* have the following attributes set
  /// - INTERFACE_TYPE
  /// - CONCRETE_TYPE
  /// - PLUGIN_NAME
  ///
  /// Plugin factory objects are grouped under the interface type name,
  /// so all factories that create the same interface are together.
  ///
  /// By adding a factory, we set the PLUGIN_ORIGIN_LIBRARY attribute to be the name
  /// of library module it was added from.
  ///
  /// A factory may fail to be added if:
  ///   * It already exists in this loader based on the combo of INTERFACE_TYPE,
  ///     CONCRETE_TYPE and PLUGIN_NAME attributes.
  ///
  /// @param fact Plugin factory object to register
  ///
  /// @return A pointer is returned to the added factory. This may be used to
  /// set
  /// additional attributes to the factory.
  ///
  /// @throws plugin_factory_missing_required_attrs
  /// One or more required attributes are not set in the given factory.
  /// @throws plugin_already_exists
  /// If the factory being added looks to already have been added before.
  ///
  /// Example:
  /// \code
  /// void add_factories( registry* pm )
  /// {
  /// plugin_factory_handle_t fact = pm->add_factory( new foo_factory() );
  /// fact->add_attribute( "file-type", "xml mit" );
  /// }
  /// \endcode
  plugin_factory_handle_t add_factory( plugin_factory* fact );

  /**
   * @brief Register a factory to generate the CONCRETE class type, specifically
   * in relation to the given INTERFACE type.
   *
   * A plugin name must also be provided. This is to succinctly describe the
   * concrete type in relation to the interface type.
   * The factory created from this will check that the given interface descends
   * from \ref pluggable and that the concrete class descends from the interface
   * class.
   *
   * Plugin factory objects are grouped under an identifier of the interface
   * type so all factories that create implementations of the same interface are
   * grouped together.
   *
   * Factories are created with the PLUGIN_ORIGIN_LIBRARY attribute set to be the
   * name of library module it was added from.
   *
   * A factory may fail to be added if:
   *   * It already exists in this loader based on the combo of INTERFACE_TYPE,
   *     CONCRETE_TYPE and PLUGIN_NAME attributes.
   *
   * This method is the primary method plugin module registration functions
   * self-register all plugins in a module.
   *
   * Example:
   *  \code
   *  void add_factories( registry* pm )
   *  {
   *  plugin_factory_handle_t fact = pm->add_factory<SomeInterface,
   * SomeDerived>( "derived" );
   *  fact->add_attribute( "file-type", "xml mit" );
   *  }
   *  \endcode
   *
   * @param plugin_name String name to describe the concrete type that
   * implements the interface type.
   *
   * @return A pointer is returned to the added factory in case
   * attributes need to be added to the factory.
   *
   * @throws plugin_already_exists
   * If the factory being added looks to already have been added before.
   *
   */
  template < typename INTERFACE, typename CONCRETE >
  plugin_factory_handle_t
  add_factory( std::string const& plugin_name )
  {
    // Call protected factory add method with the standard concrete plugin
    // factory.
    return add_factory(
      new concrete_plugin_factory< INTERFACE, CONCRETE >( plugin_name )
    );
  }

  // Alternative factory addition methods?
  // template interface/concrete, also take in "category" label?

  // Map Accessors =============================================================

  /// @brief Get map of known plugins.
  ///
  /// Get the map of all known registered plugins.
  ///
  /// @return Map of plugins
  [[nodiscard]]
  plugin_map_t const& get_plugin_map() const;

  // Deprecated? ===============================================================

  /// @brief Indicate that a module has been loaded.
  ///
  /// This method set an indication that the specified module is loaded
  /// and is used in conjunction with mark_module_as_loaded() to prevent
  /// modules from being loaded multiple times.
  ///
  /// @param name Module to indicate as loaded.
  void mark_module_as_loaded( std::string const& name );

  /// @brief Has module been loaded.
  ///
  /// This method is used to determine if the specified module has been
  /// loaded.
  ///
  /// @param name Module to indicate as loaded.
  ///
  /// @return \b true if module has been loaded. \b false otherwise.
  bool is_module_loaded( std::string const& name ) const;

  /// @brief Get list of loaded modules.
  ///
  /// This method returns a map of modules that have been marked as
  /// loaded by the mark_module_as_loaded() method.
  ///
  /// @return Map of modules loaded.
  [[nodiscard]]
  plugin_module_map_t const& get_module_map() const;

  /// @brief Name the library that is registering.
  ///
  /// The generated static registry calls this before each library's
  /// registration function; factories added while it is set record it as
  /// their PLUGIN_ORIGIN_LIBRARY.
  ///
  /// @param name Library name.
  void set_registering_library( std::string const& name );

protected:
  friend class registry_impl;  // is this needed? I clearly don't remember

  // what friend classes are.

  viame::logger_handle_t m_logger;

private:
  const std::unique_ptr< registry_impl > m_impl;
}; // end class registry

} // namespace viame

#endif
