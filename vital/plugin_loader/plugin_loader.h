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

#ifndef KWIVER_VITAL_PLUGIN_LOADER_H_
#define KWIVER_VITAL_PLUGIN_LOADER_H_

#include <vital/plugin_loader/vital_vpm_export.h>

#include <vital/vital_types.h>
#include <vital/logger/logger.h>

#include <kwiversys/DynamicLoader.hxx>

#include <vector>
#include <string>
#include <map>
#include <memory>


namespace kwiver {
namespace vital {

// base class of factory hierarchy
class plugin_factory;
class plugin_loader_filter;

using plugin_factory_handle_t = std::shared_ptr< plugin_factory >;
using plugin_factory_vector_t = std::vector< plugin_factory_handle_t >;
using plugin_map_t            = std::map< std::string, plugin_factory_vector_t >;
using plugin_module_map_t     = std::map< std::string, path_t >;
using plugin_filter_handle_t  = std::shared_ptr< plugin_loader_filter >;

class plugin_loader_impl;

// ----------------------------------------------------------------
/**
 * @brief Manages a set of plugins.
 *
 * The plugin manager keeps track of all factories from plugins that
 * are discovered on the disk.
 *
 */
class VITAL_VPM_EXPORT plugin_loader
{
public:
  typedef kwiversys::DynamicLoader   DL;

  /**
   * @brief Constructor
   *
   * This method constructs a new plugin manager.
   *
   * @param init_function Name of the plugin initialization function
   * to be called to effect loading of the plugin.
   */
  plugin_loader( std::string const& init_function,
    std::string const& shared_lib_suffix );

  virtual ~plugin_loader();

  /**
   * @brief Load all reachable plugins.
   *
   * This method loads all plugins that can be discovered on the
   * currently active search path. This method is called after all
   * search paths have been added with the add_search_path() method.
   *
   * @throws plugin_already_exists - if a duplicate plugin is detected
   */
  void load_plugins();

  /**
   * @brief Load plugins from list of directories.
   *
   * Load plugins from the specified list of directories. The
   * directories are scanned immediately and all recognized plugins
   * are loaded. The internal accumulated search path is not used for
   * this method. This is useful for adding plugins after the search
   * path has been processed.
   *
   * @param dirpath List of directories to search.
   *
   * @throws plugin_already_exists - if a duplicate plugin is detected
   */
  void load_plugins( path_list_t const& dirpath );

  /**
   * @brief Load a single plugin file.
   *
   * This method loads a single plugin file.
   *
   * @param file Name of the file to load.
   */
  void load_plugin( path_t const& file );

  /**
   * @brief Add an additional directories to search for plugins in.
   *
   * This method adds the specified directory list to the end of
   * the internal path used when loading plugins. This method can be called
   * multiple times to add multiple directories.
   *
   * Call the register_plugins() method to load plugins after you have
   * added all additional directories.
   *
   * Directory paths that don't exist will simply be ignored.
   *
   * \param dirpath Path to the directories to add to the plugin search path.
   */
  void add_search_path( path_list_t const& dirpath );

  /**
   * @brief Get plugin manager search path
   *
   *  This method returns the search path used to load algorithms.
   *
   * @return vector of paths that are searched
   */
  path_list_t const& get_search_path() const;

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
  plugin_factory_vector_t const& get_factories( std::string const& type_name ) const;

  /**
   * @brief Add factory to manager.
   *
   * This method adds the specified plugin factory to the plugin
   * manager. This method is usually called from the plugin
   * registration function in the loadable module to self-register all
   * plugins in a module.
   *
   * Plugin factory objects are grouped under the interface type name,
   * so all factories that create the same interface are together.
   *
   * @param fact Plugin factory object to register
   *
   * @return A pointer is returned to the added factory in case
   * attributes need to be added to the factory.
   *
   * @throws plugin_already_exists - if the plugin signature already has a factory.
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
   * @brief Get map of known plugins.
   *
   * Get the map of all known registered plugins.
   *
   * @return Map of plugins
   */
  plugin_map_t const& get_plugin_map() const;

  /**
   * @brief Get list of files loaded.
   *
   * This method returns the list of shared object file names that
   * successfully loaded.
   *
   * @return List of file names.
   */
  std::vector< std::string > get_file_list() const;

  /**
   * @brief Indicate that a module has been loaded.
   *
   * This method set an indication that the specified module is loaded
   * and is used in conjunction with mark_module_as_loaded() to prevent
   * modules from being loaded multiple times.
   *
   * @param name Module to indicate as loaded.
   */
  void mark_module_as_loaded( std::string const& name );

  /**
   * @brief Has module been loaded.
   *
   * This method is used to determine if the specified module has been
   * loaded.
   *
   * @param name Module to indicate as loaded.
   *
   * @return \b true if module has been loaded. \b false otherwise.
   */
  bool is_module_loaded( std::string const& name) const;

  /**
   * @brief Get list of loaded modules.
   *
   * This method returns a map of modules that have been marked as
   * loaded by the mark_module_as_loaded() method along with the name
   * of the plugin file where the call was made.
   *
   * @return Map of modules loaded and the source file.
   */
  plugin_module_map_t const& get_module_map() const;

  void clear_filters();
  void add_filter( plugin_filter_handle_t f );

protected:
  friend class plugin_loader_impl;


  kwiver::vital::logger_handle_t m_logger;

private:

  const std::unique_ptr< plugin_loader_impl > m_impl;
}; // end class plugin_loader

} } // end namespace

#endif
