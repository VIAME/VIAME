// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Interface for plugin manager.
///
/// This header exposes the following functionality:
///   - plugin_manager class (with singleton accessor)
///   - implementation_factory
///   - implementation_factory_by_name
///
/// The `plugin_manager` provides a front-end API to perform high-level actions
/// with plugins and loading.
/// These actions include:
///   - get at the singleton instance
///   - add to the search path
///   - (re)load some or all plugins based on a standard suite of module
/// locations
///   - get the plugin names of available implementations for some interface
/// type
///   - get factories for some interface type
///
/// Also provided here are some structures to facilitate the instantiation of
/// registered plugin types.
///
/// The broad, general helper is `implementation_factory` structure.
/// This base structure is flexible in what interface type to base on, as well
/// as
/// what attribute field to base implementation selection on.
/// @code{.cxx}
///  implementation_factory<MyInterface> impl_fact( plugin_factory::PLUGIN_NAME
/// );
///  std::shared_ptr<MyInterface> inst_sptr = impl_fact.create( "my_impl" );
///  assert inst_sptr != nullptr
///  inst_sptr->interface_api_method();
/// @endcode
///
/// We also provide a helper that is already setup to use the plugin name
/// attribute as the basis of select, reducing the constructor call required:
/// @code{.cxx}
///  implementation_factory_by_name<MyInterface> impl_fact;
///  std::shared_ptr<MyInterface> inst_sptr = impl_fact.create( "my_impl" );
///  assert inst_sptr != nullptr
///  inst_sptr->interface_api_method();
/// @endcode
/// This helper is likely the version to used the vast majority of the time as
/// the plugin name attribute is intended to be the primary selection criterion,
/// as well as is constrained to be unique upon registration per interface
/// category.

#ifndef KWIVER_VITAL_PLUGIN_MANAGER_H
#define KWIVER_VITAL_PLUGIN_MANAGER_H

#include <viame/algorithm_framework/plugin/vital_vpm_export.h>

#include <viame/core_types/bitflags.h>
#include <viame/algorithm_framework/exceptions/plugin.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/core_types/noncopyable.h>
#include <viame/algorithm_framework/plugin/plugin_factory.h>
#include <viame/algorithm_framework/plugin/registry.h>
#include <viame/algorithm_framework/util/demangle.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

namespace kwiver::vital {

// ----------------------------------------------------------------------------
/// @brief Vital plugin manager.
///
/// This class is the main plugin manager for all kwiver components.
///
/// Behaves as a decorator for registry
class VITAL_VPM_EXPORT plugin_manager
  : private kwiver::vital::noncopyable
{
public:
  typedef std::string module_t; // module name type

  enum class plugin_type
  {
    PROCESSES  = 0x0001,
    ALGORITHMS = 0x0002,
    APPLETS    = 0x0004,
    // TODO: Are explorer/others/legacy of use anywhere?
    EXPLORER   = 0x0008,
    OTHERS     = 0x0020,
    LEGACY     = 0x0040,
    DEFAULT    = 0x00f7,  // sum of the above.
    ALL        = 0xffff,
  };

  KWIVER_DECLARE_BITFLAGS( plugin_types, plugin_type );

  static plugin_manager& instance();  // singleton interface

// Loading Plugins -----------------------------------------------------------

/// @brief Load all reachable plugins.
///
/// This method calls the registration function of every library VIAME
/// was built with whose kind is named in \p types -- see
/// add_static_registrar. There is nothing to discover and no path to
/// search: the set is the build's.
///
/// A kind that has already been loaded is skipped, so calling this twice
/// registers nothing the second time. To start over, call
/// reload_all_plugins().
///
/// @throws plugin_already_exists - if a duplicate plugin is detected
  void load_all_plugins( plugin_types types = plugin_type::DEFAULT );

/// @brief Add a registration function that is called instead of being found.
///
/// A library compiled with `viame_register_statically` has its registration
/// function linked in rather than left in a loadable module, and hands it
/// here as it loads. `load_all_plugins` calls every registrar added this way,
/// in the order they were added.
///
/// It is called with the same `plugin_types` mask the caller gave
/// `load_all_plugins`, and registers only what the mask asks for.
///
/// Adding the same function twice adds it twice; registration functions
/// guard on their own module name, so the second call does nothing.
///
/// @param registrar Function to call with the loader.
  static void add_static_registrar(
    void ( *registrar )( registry&, plugin_types ) );

/// @brief Add factory to manager.
///
/// This method adds the specified plugin factory to the plugin
/// manager. This method is usually called from the plugin
/// registration function in the loadable module to self-register all
/// plugins in a module.
///
/// The plugin_manager takes ownership of the factory object supplied
/// and deletes it when the program terminates. Therefore the factory
/// object must be allocated from the heap and never allocated on the
/// stack.
///
/// Plugin factory objects are grouped under the interface type name,
/// so all factories that create the same interface are together.
///
/// @param fact Plugin factory object to register
///
/// @return A pointer is returned to the added factory so attributes
/// can to be added to the factory.
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

// Getting at Factories ------------------------------------------------------

/// @brief Get list of factories for interface type.
///
/// This method returns a list of pointer to factory methods that
/// create objects of the desired interface type.
///
/// @tparam T Type of the interface required
///
/// @return Vector of factories. (vector may be empty)
  template < class INTERFACE >
  [[nodiscard]]

  plugin_factory_vector_t const&
  get_factories()
  {
    return get_factories( get_interface_name< INTERFACE >() );
  }

/// @brief Reload all plugins.
///
/// The current list of factories is deleted, all currently open
/// files are closed, and storage released. The module loading
/// process is performed using the current state of this manager.
///
/// This effectively resets the singleton.
  void reload_all_plugins();

/**
 * @brief Get the vector of plugin implementation names registered for a given
 * interface type.
 *
 * If a registered plugin does not have a non-empty PLUGIN_NAME, we will
 * represent those here as "<UNNAMED>" entries.
 *
 * The returned vector may be empty if there are no implementations currently
 * registered for the given interface.
 *
 * @tparam INTERFACE Type of the interface to check for plugin implementations
 * of.
 *
 * @return Vector of plugin implementation names.
 */
  template < typename INTERFACE >
  [[nodiscard]]

  std::vector< std::string >
  impl_names() const
  {
    return _impl_names( get_interface_name< INTERFACE >() );
  }

/// @brief Has the module been loaded
///
/// This method reports if the specified module has been loaded.
///
/// @return \b true if module has been loaded. \b false otherwise.
  bool is_module_loaded( module_t const& name ) const;

// Deprecated? ---------------------------------------------------------------

/// @brief Mark module as loaded.
///
/// This method adds the specified module name to the list of loaded
/// modules. The presence of a module name can be determined with the
/// is_module_loaded() method.
///
/// @param name Module to mark as loaded.
  void mark_module_as_loaded( module_t const& name );

protected:
// Deprecated? ---------------------------------------------------------------
// Some of these are used by the explorer via the "internal" subclass.
// Maybe reinstate or maybe determine alternative to having an "internal"
// subclass impl to expose what is needed.
// Some of it seemed to be access to things registered under certain
// categories?

  registry* get_registry();

/// @brief Get map of known plugins.
///
/// Get the map of all known registered plugins.
///
/// @return Map of plugins
  plugin_map_t const& plugin_map();

/// @brief Get list of loaded modules
///
/// This call returns a map of loaded modules with the files they
/// were defined in.
///
/// @return Map of loaded modules.
  std::map< std::string, std::string > const& module_map() const;

// Protected constructor/destructor to enforce use of singleton instance.
  plugin_manager();
  ~plugin_manager();

public:
// Factory Stuff -------------------------------------------------------------
/// @brief Get list of factories for interface type.
///
/// This method returns a list of pointer to factory methods that
/// create objects of the desired interface type.
///
/// The input here is expected to be the interface name of a type T as returned
/// from `get_interface_name<T>()`, which should be a human-readable string that
/// has defined.
/// For clarity, we consider a class an "interface type" as a class type that
/// defines one or more pure-virtual functions for derived classes to
/// implement.
///
/// This method is private so as to not expose this for arbitrary use, but only
/// for the public, templated method defined above to call this appropriately.
///
/// @param type_name Type name of the interface required
///
/// @return Vector of factories. (vector may be empty)
  plugin_factory_vector_t const& get_factories( std::string const& type_name );

/**
 * @brief Protected accessor of interface implementation names given the
 * interface type name.
 *
 * The returned vector may be empty if there are no implementations currently
 * registered for the given interface.
 *
 * @param interface_type_name String type name of the interface to check for
 * plugin implementations of.
 *
 * @return Vector of plugin implementation names.
 */
  [[nodiscard]]
  std::vector< std::string > _impl_names(
    std::string const& interface_type_name ) const;

private:
/// @brief Get logger handle.
///
/// This method returns the handle for the plugin manager
/// logger. This logger can be used by the plugin module to log
/// diagnostics during the factory creation process.
///
/// @return Handle to plugin_manager logger
  logger_handle_t logger();

  class priv;

  const std::unique_ptr< priv > m_priv;

  static plugin_manager* s_instance;
}; // end class plugin_manager

KWIVER_DECLARE_OPERATORS_FOR_BITFLAGS( plugin_manager::plugin_types )

// ----------------------------------------------------------------------------
/// \brief Typed implementation factory.
///
/// This struct implements a typed implementation factory. It uses the
/// \ref plugin_manager to create an instance of a class that
/// creates a specific variant of the interface type.
///
/// The list of factories that create variants for the specified
/// interface type is queried from the \ref plugin_manager. This list
/// is searched for an entry that has the desired value in the
/// specified factory attribute.
///
/// This struct is intended as a base class with derived structs
/// specifying the desired attribute name, as in \ref
/// implementation_factory_by_name.
///
/// \tparam I Interface type that is created
// ----------------------------------------------------------------------------
template < typename I >
class implementation_factory
{
public:
  /// @brief CTOR
  ///
  /// This constructor creates an implementation factory that uses a
  /// specific attribute to chose the factory object. The name of the
  /// attribute is supplied in this call is used as the key field. The
  /// create() selects the factory which has a specific value in this
  /// field.
  ///
  /// @param attr Name of attribute to use as key field.
  explicit implementation_factory( std::string attr )
    : m_attr( std::move( attr ) )
  {}

  /// @brief Find object factory based on attribute value.
  ///
  /// This form is used when it is desired to get at an implementation's factory
  /// in order to access other factory attributes of that plugin, e.g. it's
  /// description.
  ///
  /// @param attr Attribute value string.
  ///
  /// @return Address of the factory object for the templated type with
  /// the specified attribute value.
  ///
  /// @throws kwiver::vital::plugin_factory_not_found
  plugin_factory_handle_t
  find_factory( const std::string& value )
  {
    // Get singleton plugin manager
    kwiver::vital::plugin_manager& pm =
      kwiver::vital::plugin_manager::instance();

    auto fact_list = pm.get_factories< I >();
    // Scan fact_list for factory to instantiate from.
    for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
    {
      std::string attr_val;
      if( a_fact->get_attribute( m_attr, attr_val ) && ( attr_val == value ) )
      {
        return a_fact;
      }
    } // end foreach

    std::stringstream str;
    str << "Could not find factory where attr \"" << m_attr << "\" is \"" <<
      value
        << "\" for interface type \"" << get_interface_name< I >()
        << "\"";

    VITAL_THROW( kwiver::vital::plugin_factory_not_found, str.str() );
  }

  /// @brief Create object based on attribute value.
  ///
  /// The list of factories which create the interface type I is
  /// scanned for an entry which contains the supplied value in the
  /// attribute field. When one is found, that factory is used to
  /// create a new object. An exception is thrown if the attribute
  /// field is not present or no factory has the requested value.
  ///
  /// @param value Attribute value.
  ///
  /// @return Pointer to new object of type I.
  ///
  /// @throws kwiver::vital::plugin_factory_not_found
  std::shared_ptr< I >
  create( const std::string& value, config_block_sptr const cb )
  {
    plugin_factory_handle_t a_fact = this->find_factory( value );
    return std::dynamic_pointer_cast< I >( a_fact->from_config( cb ) );
  }

private:
  // member data
  std::string m_attr; // Name of the attribute
};

// ----------------------------------------------------------------------------
/// @brief Implementation factory that uses name attribute.
///
/// This struct provides a common implementation for creating objects
/// of a specific type based on the "name" attribute.
///
/// Example usage:
/// \code
/// // create name for factory to create specific interface object.
/// typedef kwiver::vital::implementation_factory_by_name<
/// sprokit::process_instrumentation > instrumentation_factory;
///
/// // instantiate factory class when needed.
/// instrumentation_factory ifact;
/// auto instr = ifact.create( provider );
/// \endcode
///
/// \throws plugin_factory_not_found
template < typename T >
class implementation_factory_by_name
  : public implementation_factory< T >
{
public:
  implementation_factory_by_name()
    : implementation_factory< T >( kwiver::vital::plugin_factory::PLUGIN_NAME )
  {}
};

} // end namespace

#endif // KWIVER_VITAL_PLUGIN_MANAGER_H
