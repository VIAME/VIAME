/*ckwg +29
 * Copyright 2016, 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Interface for plugin manager.
 */

#ifndef KWIVER_VITAL_PLUGIN_MANAGER_H
#define KWIVER_VITAL_PLUGIN_MANAGER_H

#include <vital/plugin_loader/vital_vpm_export.h>

#include <vital/plugin_loader/plugin_loader.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/exceptions/plugin.h>
#include <vital/logger/logger.h>
#include <vital/util/demangle.h>
#include <vital/noncopyable.h>

#include <memory>
#include <sstream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Vital plugin manager.
 *
 * This class is the main plugin manager for all kwiver components.
 *
 * Behaves as a decorator for plugin_loader
 */
class VITAL_VPM_EXPORT plugin_manager
  : private kwiver::vital::noncopyable
{
public:
  typedef std::string module_t; // module name type

  enum class plugin_type
  {
    PROCESSES               = 0x0001,
    ALGORITHMS              = 0x0002,
    APPLETS                 = 0x0004,
    EXPLORER                = 0x0008,
    PROCESS_INSTRUMENTATION = 0x0010,
    OTHERS                  = 0x0020,
    LEGACY                  = 0x0040,
    ALL                     = 0xffff
  };


  static plugin_manager& instance();  // singleton interface

  /**
   * @brief Load all reachable plugins.
   *
   * This method loads all plugins that can be discovered on the
   * currently active search path. This method is called after all
   * search paths have been added with the add_search_path() method.
   *
   * The first call to this method will load all known
   * plugins. Subsequent calls will not load anything. If the plugins
   * need to be reloaded, call the reload_plugins() method. if an
   * additional directory list must be scanned after plugins are
   * loaded, call load_plugins() with a list of directories to add
   * more plugins to the manager.
   *
   * @throws plugin_already_exists - if a duplicate plugin is detected
   */
  void load_all_plugins( plugin_type type = plugin_type::ALL );

  /**
   * @brief Load plugins from list of directories.
   *
   * Load plugins from the specified list of directories. The
   * directories are scanned immediately and all recognized plugins
   * are loaded.
   *
   * @param dirpath List of directories to search.
   *
   * @throws plugin_already_exists - if a duplicate plugin is detected
   */
  void load_plugins( path_list_t const& dirpath );

  /**
   * @brief Add an additional directories to search for plugins in.
   *
   * This method adds the specified directory list to the end of the
   * internal path used when loading plugins. This method can be
   * called multiple times to add multiple sets of directories. Each
   * directory is separated from the next by the standard system path
   * separator character.
   *
   * Single directories can be added with this method.
   *
   * Call the load_plugins() method to load plugins after you have
   * added all additional directories.
   *
   * Directory paths that don't exist will simply be ignored.
   *
   * \param dirpath Path to the directories to add to the plugin search path.
   */
  void add_search_path(path_t const& dirpath);

  /**
   * @brief Add an additional directories to search for plugins in.
   *
   * This method adds the specified directory list to the end of the
   * internal path used when loading plugins. This method can be
   * called multiple times to add multiple sets of directories.
   *
   * Call the load_plugins() method to load plugins after you have
   * added all additional directories.
   *
   * Directory paths that don't exist will simply be ignored.
   *
   * \param dirpath Path to the directories to add to the plugin search path.
   */
  void add_search_path( path_list_t const& dirpath );

  /**
   * @brief Add factory to manager.
   *
   * This method adds the specified plugin factory to the plugin
   * manager. This method is usually called from the plugin
   * registration function in the loadable module to self-register all
   * plugins in a module.
   *
   * The plugin_manager takes ownership of the factory object supplied
   * and deletes it when the program terminates. Therefore the factory
   * object must be allocated from the heap and never allocated on the
   * stack.
   *
   * Plugin factory objects are grouped under the interface type name,
   * so all factories that create the same interface are together.
   *
   * @param fact Plugin factory object to register
   *
   * @return A pointer is returned to the added factory so attributes
   * can to be added to the factory.
   *
   * Example:
   \code
   void add_factories( plugin_loader* pm )
   {
     plugin_factory_handle_t fact = pm->add_factory( new foo_factory() );
     fact->add_attribute( "file-type", "xml mit" );
   }
   \endcode
   */
  plugin_factory_handle_t add_factory( plugin_factory* fact );

  /**
   * @brief Get list of factories for interface type.
   *
   * This method returns a list of pointer to factory methods that
   * create objects of the desired interface type.
   *
   * @param type_name Type name of the interface required
   *
   * @return Vector of factories. (vector may be empty)
   */
  plugin_factory_vector_t const& get_factories( std::string const& type_name );

  /**
   * @brief Get list of factories for interface type.
   *
   * This method returns a list of pointer to factory methods that
   * create objects of the desired interface type.
   *
   * @tparam T Type of the interface required
   *
   * @return Vector of factories. (vector may be empty)
   */
  template <class T>
  plugin_factory_vector_t const& get_factories()
  {
    return get_factories (typeid( T ).name() );
  }

  /**
   * @brief Reload all plugins.
   *
   * The current list of factories is deleted, all currently open
   * files are closed, and storage released. The module loading
   * process is performed using the current state of this manager.
   *
   * This effectively resets the singleton.
   */
  void reload_plugins();

  /**
   * @brief Has the module been loaded
   *
   * This method reports if the specified module has been loaded.
   *
   * @return \b true if module has been loaded. \b false otherwise.
   */
  bool is_module_loaded( module_t const& name) const;

  /**
   * @brief Mark module as loaded.
   *
   * This method adds the specified module name to the list of loaded
   * modules. The presence of a module name can be determined with the
   * is_module_loaded() method.
   *
   * @param name Module to mark as loaded.
   */
  void mark_module_as_loaded( module_t const& name );

  /**
   * @brief Add path from environment variable name.
   *
   * This method adds the path from the environment variable to the end
   * of the current search path.
   *
   * @param env_var Name of environment variable.
   */
  void add_path_from_environment( std::string env_var);

protected:

  plugin_loader* get_loader();

  /**
   * @brief Get list of files loaded.
   *
   * This method returns the list of shared object file names that
   * successfully loaded.
   *
   * @return List of file names.
   */
  std::vector< std::string > file_list();

  /**
   * @brief Get map of known plugins.
   *
   * Get the map of all known registered plugins.
   *
   * @return Map of plugins
   */
  plugin_map_t const& plugin_map();

  /**
   * @brief Get list of loaded modules
   *
   * This call returns a map of loaded modules with the files they
   * were defined in.
   *
   * @return Map of loaded modules.
   */
  std::map< std::string, std::string > const& module_map() const;

  /**
   * @brief Get plugin manager search path
   *
   *  This method returns the search path used to load algorithms.
   *
   * @return vector of paths that are searched
   */
  path_list_t const& search_path() const;


  plugin_manager();
  ~plugin_manager();

private:

  /**
   * @brief Get logger handle.
   *
   * This method returns the handle for the plugin manager
   * logger. This logger can be used by the plugin module to log
   * diagnostics during the factory creation process.
   *
   * @return Handle to plugin_manager logger
   */
  logger_handle_t logger();

  class priv;
  const std::unique_ptr< priv > m_priv;

  static plugin_manager* s_instance;

}; // end class plugin_manager


// ==================================================================
/**
 * \brief Typed implementation factory.
 *
 * This struct implements a typed implementation factory. It uses the
 * \ref plugin_manager to create an instance of a class that
 * creates a specific variant of the interface type.
 *
 * The list of factories that create variants for the specified
 * interface type is queried from the \ref plugin_manager. This list
 * is searched for an entry that has the desired value in the
 * specified factory attribute.
 *
 * This struct is intended as a base class with derived structs
 * specifying the desired attribute name, as in \ref
 * implementation_factory_by_name.
 *
 *
 * \tparam I Interface type that is created
 */
template <typename I>
class implementation_factory
{
public:
  /**
   * @brief CTOR
   *
   * This constructor creates an implementation factory that uses a
   * specific attribute to chose the factory object. The name of the
   * attribute is supplied in this call is used as the key field. The
   * create() selects the factory which has a specific value in this
   * field.
   *
   * @param attr Name of attribute to use as key field.
   */
  implementation_factory( std::string const& attr)
    : m_attr( attr)
  { }

  /**
   * @brief Find object factory based on attribute value.
   *
   * @param attr Attribute value string.
   *
   * @return Address of the factory object for the templated type with
   * the specified attribute value.
   *
   * @throws kwiver::vital::plugin_factory_not_found
   */
  plugin_factory_handle_t find_factory( const std::string& value )
  {
    // Get singleton plugin manager
    kwiver::vital::plugin_manager& pm = kwiver::vital::plugin_manager::instance();

    auto fact_list = pm.get_factories( typeid( I ).name() );
    // Scan fact_list for CONCRETE_TYPE
    for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
    {
      std::string attr_val;
      if ( a_fact->get_attribute( m_attr, attr_val ) && ( attr_val == value ) )
      {
        return a_fact;
      }
    } // end foreach

    std::stringstream str;
    str << "Could not find factory where attr \"" << m_attr << "\" is \"" << value
        << "\" for interface type \"" << demangle( typeid(I).name() )
        << "\"";

    VITAL_THROW( kwiver::vital::plugin_factory_not_found, str.str() );
  }

  /**
   * @brief Create object based on attribute value.
   *
   * The list of factories which create the interface type I is
   * scanned for an entry which contains the supplied value in the
   * attribute field. When one is found, that factory is used to
   * create a new object. An exception is thrown if the attribute
   * field is not present or no factory has the requested value.
   *
   * @param value Attribute value.
   *
   * @return Pointer to new object of type I.
   *
   * @throws kwiver::vital::plugin_factory_not_found
   */
  I* create( const std::string& value )
  {
    plugin_factory_handle_t a_fact = this->find_factory( value );
    return a_fact->create_object<I>();
  }

private:
  // member data
  std::string m_attr; // Name of the attribute
};


// ----------------------------------------------------------------
/**
 * @brief Implementation factory that uses name attribute.
 *
 * This struct provides a common implementation for creating objects
 * of a specific type based on the "name" attribute.
 *
 * Example usage:
 * \code
// create name for factory to create specific interface object.
typedef kwiver::vital::implementation_factory_by_name< sprokit::process_instrumentation > instrumentation_factory;

// instantiate factory class when needed.
instrumentation_factory ifact;
auto instr = ifact.create( provider );
\endcode
 *
 * \throws plugin_factory_not_found
 */
template <typename T>
class implementation_factory_by_name
  : public implementation_factory< T >
{
public:
  implementation_factory_by_name()
    : implementation_factory<T>( kwiver::vital::plugin_factory::PLUGIN_NAME )
  { }
};

} } // end namespace

#endif /* KWIVER_VITAL_PLUGIN_MANAGER_H */
