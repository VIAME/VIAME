// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PLUGIN_LOADER_H_
#define KWIVER_VITAL_PLUGIN_LOADER_H_

#include <viame/algorithm_framework/plugin/vital_vpm_export.h>

#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/core_types/vital_types.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace kwiver::vital {

// base class of factory hierarchy
class plugin_factory;
class plugin_loader_filter;

using plugin_factory_handle_t = std::shared_ptr< plugin_factory >;
using plugin_factory_vector_t = std::vector< plugin_factory_handle_t >;
using plugin_map_t            = std::map< std::string,
  plugin_factory_vector_t >;
using plugin_module_map_t     = std::map< std::string, path_t >;
using plugin_filter_handle_t  = std::shared_ptr< plugin_loader_filter >;

class plugin_loader_impl;

/**
 * @brief Manage dynamically loading plugin modules from search paths given a
 *    known "initialization" function to run.
 *
 * The plugin manager keeps track of all factories from plugins that
 * are discovered on the disk.
 *
 */
class VITAL_VPM_EXPORT plugin_loader
{
public:
  /**
   * @brief Constructor
   *
   * @param init_function Name of the plugin initialization function
   *    to be called to effect loading of the plugin.
   * @param shared_lib_suffix Shared library suffix string for the platform
   *    being loaded from.
   */
  plugin_loader(
    std::string const& init_function,
    std::string const& shared_lib_suffix );

  virtual ~plugin_loader();

  /// @brief Load all reachable plugins.
  ///
  /// This method loads all plugins that can be discovered on the
  /// currently active search path. This method is called after all
  /// search paths have been added with the add_search_path() method.
  ///
  /// @throws plugin_already_exists - if a duplicate plugin is detected
  void load_plugins();

  /// @brief Load plugins from list of directories.
  ///
  /// Load plugins from the specified list of directories. The
  /// directories are scanned immediately and all recognized plugins
  /// are loaded. The internal accumulated search path is not used for
  /// this method. This is useful for adding plugins after the search
  /// path has been processed.
  ///
  /// @param dirpath List of directories to search.
  ///
  /// @throws plugin_already_exists - if a duplicate plugin is detected
  void load_plugins( path_list_t const& dirpath );

  /// @brief Load a single plugin file.
  ///
  /// This method loads a single plugin file.
  ///
  /// @param file Name of the file to load.
  void load_plugin( path_t const& file );

  // Search Stuff --------------------------------------------------------------
  /// @brief Add an additional directories to search for plugins in.
  ///
  /// This method adds the specified directory list to the end of
  /// the internal path used when loading plugins. This method can be called
  /// multiple times to add multiple directories.
  ///
  /// Call the register_plugins() method to load plugins after you have
  /// added all additional directories.
  ///
  /// Directory paths that don't exist will simply be ignored.
  ///
  /// \param dirpath Path to the directories to add to the plugin search path.
  void add_search_path( path_list_t const& dirpath );

  /// @brief Get plugin manager search path
  ///
  ///  This method returns the search path used to load algorithms.
  ///
  /// @return vector of paths that are searched
  path_list_t const& get_search_path() const;

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
  /// By adding a factory, we set the PLUGIN_FILE_NAME attribute to be the name
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
  /// void add_factories( plugin_loader* pm )
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
   * Factories are created with the PLUGIN_FILE_NAME attribute set to be the
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
   *  void add_factories( plugin_loader* pm )
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

  // TODO: This doesn't seem to be used anywhere
  /// @brief Get list of files loaded.
  ///
  /// This method returns the list of shared object file names that
  /// successfully loaded.
  ///
  /// @return List of file names.
  std::vector< std::string > get_file_list() const;

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
  /// loaded by the mark_module_as_loaded() method along with the name
  /// of the plugin file where the call was made.
  ///
  /// @return Map of modules loaded and the source file.
  [[nodiscard]]
  plugin_module_map_t const& get_module_map() const;

  void clear_filters();
  void add_filter( plugin_filter_handle_t f );

protected:
  friend class plugin_loader_impl;  // is this needed? I clearly don't remember

  // what friend classes are.

  kwiver::vital::logger_handle_t m_logger;

private:
  const std::unique_ptr< plugin_loader_impl > m_impl;
}; // end class plugin_loader

} // end namespace

#endif
