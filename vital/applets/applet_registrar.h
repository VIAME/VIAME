// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
