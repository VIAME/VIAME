/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#ifndef KWIVER_VITAL_PLUGIN_LOADER_FILTER_H
#define KWIVER_VITAL_PLUGIN_LOADER_FILTER_H

#include <kwiversys/DynamicLoader.hxx>

#include <vital/vital_types.h>

#include <memory>

namespace kwiver {
namespace vital {

// base class of factory hierarchy
class plugin_factory;
class plugin_loader;

using plugin_factory_handle_t = std::shared_ptr< plugin_factory >;

// -----------------------------------------------------------------
/** Interface to plugin loader filters.
 *
 *
 */
class plugin_loader_filter
{
public:
  using DL = kwiversys::DynamicLoader;

  // -- CONSTRUCTORS --
  plugin_loader_filter() = default;
  virtual ~plugin_loader_filter() = default;

  /**
   * @brief Test if plugin should be loaded.
   *
   * This method is a hook that can be implemented by a derived class
   * to verify that the specified plugin should be loaded. This
   * provides an application level approach to filter specific plugins
   * from a directory. The default implementation is to load all
   * discovered plugins.
   *
   * This method is called after the plugin is opened and the
   * designated initialization method has been located but not yet
   * called. Returning \b false from this method will result in the
   * library being closed without further processing.
   *
   * The library handle can be used to inspect the contents of the
   * plugin as needed.
   *
   * @param path File path to the plugin being loaded.
   * @param lib_handle Handle to library.
   *
   * @return \b true if the plugin should be loaded, \b false if plugin should not be loaded
   */
  virtual bool load_plugin( path_t const& path,
                            DL::LibraryHandle lib_handle ) const
    { return true; }

  /**
   * @brief Test if factory should be registered.
   *
   * This method is a hook that can be implemented by a derived class
   * to verify that the specified factory should be registered. This
   * provides an application level approach to filtering specific
   * class factories from a plugin.
   *
   * This method is called as the plugin is registering class
   * factories and can inspect attributes to determine if this factory
   * should be registered. Returning \b false will prevent this
   * factory from being registered with the plugin manager.
   *
   * A slight misapplication of this hook method could be to add
   * specific attributes to a set of factories before they are
   * registered.
   *
   * @param fact Pointer to the factory object.
   *
   * @return \b true if the plugin should be registered, \b false otherwise.
   */
  virtual bool add_factory( plugin_factory_handle_t fact ) const
    { return true; }


  // reference to the owning loader.
  plugin_loader* m_loader;

}; // end class plugin_loader_filter

using plugin_filter_handle_t = std::shared_ptr< plugin_loader_filter >;

} } // end namespace

#endif //KWIVER_VITAL_PLUGIN_LOADER_FILTER_H
