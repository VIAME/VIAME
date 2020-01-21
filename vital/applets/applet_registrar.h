/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#ifndef KWIVER_TOOLS_KWIVER_APPLET_REGISTER_H
#define KWIVER_TOOLS_KWIVER_APPLET_REGISTER_H

#include <vital/plugin_loader/plugin_registrar.h>
#include <vital/applets/kwiver_applet.h>

namespace kwiver {

/// Registrar class for applets
/**
 * This class implements the specific process for registering
 * applets. The plugin name and description is taken from the plugin
 * definition.
 *
 */
class applet_registrar
  : public plugin_registrar
{
public:
  applet_registrar( kwiver::vital::plugin_loader& vpl,
                    const std::string& mod_name )
    : plugin_registrar( vpl, mod_name )
  {
  }


  // ----------------------------------------------------------------------------
  /// Register a tool plugin.
  /**
   * A tool applet of the specified type is registered with the plugin
   * manager. All plugins of this type are marked with the applet
   * category.
   *
   * @tparam tool_t Type of the tool being registered.
   *
   * @return The plugin loader reference is returned.
   */
  template <typename tool_t>
  kwiver::vital::plugin_factory_handle_t register_tool()
  {
    using kvpf = kwiver::vital::plugin_factory;

    kwiver::vital::plugin_factory* fact = new kwiver::vital::plugin_factory_0< tool_t >(
      typeid( kwiver::tools::kwiver_applet ).name() );

    fact->add_attribute( kvpf::PLUGIN_NAME,      tool_t::_plugin_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,  tool_t::_plugin_description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,  this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION, this->organization() )
      .add_attribute( kvpf::PLUGIN_CATEGORY,     kvpf::APPLET_CATEGORY )
      ;

    return plugin_loader().add_factory( fact );
  }
};

} // end namespace

#endif /* KWIVER_TOOLS_KWIVER_APPLET_REGISTER_H */
