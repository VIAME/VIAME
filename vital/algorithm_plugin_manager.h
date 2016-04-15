/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Singleton manager of plug-in stuff
 */

#ifndef VITAL_ALGORITHM_PLUGIN_MANAGER_H_
#define VITAL_ALGORITHM_PLUGIN_MANAGER_H_

#include <string>
#include <vector>
#include <memory>


#include <vital/noncopyable.h>

#include <vital/vital_apm_export.h>
#include <vital/vital_types.h>

namespace kwiver {
namespace vital {


/**
 * Plugin Manager for algorithm implementation extensions
 */
class VITAL_APM_EXPORT algorithm_plugin_manager
  : private kwiver::vital::noncopyable
{
public:
  /// Get the reference to the singleton instance of this class
  static algorithm_plugin_manager& instance();

  /// Load all plugins on first call.
  /**
   * This static method loads all plugins on the first call and does
   * nothing on all subsequent calls. This is designed to load plugins
   * in a concurrent application where the first thread to start is
   * non-deterministic. All threads would call this method on starting
   * and the first one that completes has loaded all plugins and the
   * other callers will return.
   *
   * If you must reload plugins after this method has been called, use
   * the register_plugins() method.
   *
   * If the singleton has not been created prior to this call, is is
   * created by this call.
   *
   * @return \b true if plugins were loaded, \b false if plugins were
   * already loaded.
   */
  static bool load_plugins_once();

  /// (Re)Load plugin libraries found along current search paths
  /**
   * This method loads or reloads plugins. This method must be called
   * to get the plugins since this is not done by the CTOR.
   *
   * In the case of reloading plugins, existing plugins will be
   * replaced with newer ones.
   *
   * \param name If a name is provided, we will only load plugins whose name
   *             corresponds to the name given. If no plugins with the given
   *             name are found, nothing is loaded.
   */
  void register_plugins( std::string name = std::string() );

  /// Add an additional directory to search for plugins in.
  /**
   * This method adds the specified directory to the list used when
   * loading plugins. This method can be called multiple times to add
   * multiple directories. Call the register_plugins() method to load
   * plugins after you have added all additional directories.
   *
   * Directory paths that don't exist will simply be ignored.
   *
   * \param dirpath Path to the directory to add to the plugin search path
   */
  void add_search_path(path_t dirpath);

  /// Get plugin manager search path
  /**
   *  This method returns the search path used to load algorithms.
   *
   * @return vector of paths that are searched
   */
  std::vector< std::string > const& get_search_path() const;

  /// Get the list currently registered module names.
  /**
   * A module's name is defined as the filename minus the standard platform
   * module library suffix. For example, on Windows, if a module library was
   * named ``vital_foo.dll``, the module's name would be "vital_foo". Similarly
   * on a unix system, ``vital_bar.so`` would have the name "vital_bar".
   */
  std::vector< std::string > registered_module_names() const;

private:
  class impl;
  const std::unique_ptr< impl > m_impl;

  /// Private constructor
  /**
   * The singleton instance of this class should only be accessed via the
   * ``instance()`` static method.
   */
  algorithm_plugin_manager();

  /// private deconstructor (singleton)
  virtual ~algorithm_plugin_manager();

  static algorithm_plugin_manager* s_instance;
};

} } // end namespace

#endif // VITAL_ALGORITHM_PLUGIN_MANAGER_H_
