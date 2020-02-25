/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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

#ifndef PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_
#define PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_

#include <sprokit/processes/adapters/kwiver_adapter_export.h>

#include <sprokit/pipeline/pipeline.h>
#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_info.h>
#include <vital/plugin_loader/plugin_registrar.h>

#include <memory>

namespace kwiver {

// ----------------------------------------------------------------
/**
 * @brief Base class for embedded pipeline extension
 *
 */
class KWIVER_ADAPTER_EXPORT embedded_pipeline_extension
{
public:

  /* @brief Context passed to loaded extension
   *
   * This class provides access to the main embedded pipeline context
   * and is supplied to the plugin.
   */
  class context
  {
  public:
    virtual ~context() = default;

    // Returns pointer to pipeline object
    virtual sprokit::pipeline_t pipeline() = 0;

    // Returns a logger handle
    virtual vital::logger_handle_t logger() = 0;

    // Returns the whole pipeline config
    virtual kwiver::vital::config_block_sptr pipe_config() = 0;
  };


  // -- CONSTRUCTORS --
  virtual ~embedded_pipeline_extension() = default;

  /**
   * @brief pipeline pre-setup hook
   *
   * This method is called before the pipeline is setup. A context
   * object is supplied to this hook so it can query its running environment.
   *
   * @param ctxt The calling context.
   */
  virtual void pre_setup( context& ctxt ) { };

  /**
   * @brief pipeline post-setup hook
   *
   * This method is called after the pipeline is setup. A context
   * object is supplied to this hook so it can query its running environment.
   *
   * @param ctxt The calling context.
   */
  virtual void post_setup( context& ctxt ) { };

  /**
   * @brief End of data received from pipeline.
   *
   * This method is called when the end of data marker is received
   * from the pipeline output adapter via a receive() call. If the
   * pipeline has a sink process and does not contain an
   * output_adapter, then this method will never be called.
   *
   * @param ctxt The calling context
   */
  virtual void end_of_output( context& ctxt ) { };

  /**
   * @brief Configure provider.
   *
   * This method sends the epx config sub-block to the
   * implementation. The derived class would use the contents of this
   * config block to modify its behaviour. This is how the epx gets
   * its configuration and only needs to be overridden if the epx is
   * expecting config items.
   *
   * @param conf Configuration block.
   */
  virtual void configure( kwiver::vital::config_block_sptr const conf );

  /**
   * @brief Get default configuration block.
   *
   * This method returns the default configuration block for this
   * pipeline extension and should contain all configuration items
   * that are needed by this implementation. The config block returned
   * is used during introspection to provide documentation on what
   * config parameters are needed and what they mean. The config block
   * should contain any default values for the config items.
   *
   * @return Pointer to config block.
   */
  virtual kwiver::vital::config_block_sptr get_configuration() const;

protected:
  embedded_pipeline_extension();

}; // end class embedded_pipeline_extension

// define pointer type for this interface
using embedded_pipeline_extension_sptr = std::shared_ptr< embedded_pipeline_extension >;


// ============================================================================
/// Derived class to register Embedded Pipeline Extensions
/**
 * Embedded Pipeline Extension Registrar
 *
 * This class assists in registering embedded pipeline extensions
 *
 */
class embedded_pipeline_extension_registrar
  : public plugin_registrar
{
public:
  embedded_pipeline_extension_registrar( kwiver::vital::plugin_loader& vpl,
                    const std::string& mod_name )
    : plugin_registrar( vpl, mod_name )
  {
  }

    // Use forced naming convention for modules
  bool is_module_loaded() override
  {
    return plugin_loader().is_module_loaded( "epx." + module_name() );
  }

  void mark_module_as_loaded() override
  {
    plugin_loader().mark_module_as_loaded( "epx." + module_name() );
  }

  // ----------------------------------------------------------------------------
  /// Register an Embedded Pipeline Extension plugin.
  /**
   * A EPX of the specified type is registered with the plugin
   * manager.
   *
   * @tparam epx_t Type of the EPX being registered.
   *
   * @return The plugin loader reference is returned.
   */
  template <typename epx_t>
  kwiver::vital::plugin_factory_handle_t register_EPX()
  {
    using kvpf = kwiver::vital::plugin_factory;

    kwiver::vital::plugin_factory* fact = new kwiver::vital::plugin_factory_0< epx_t >(
      typeid( kwiver::embedded_pipeline_extension ).name() );

    fact->add_attribute( kvpf::PLUGIN_NAME,      epx_t::_plugin_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,  epx_t::_plugin_description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,  this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION, this->organization() )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, "embedded-pipeline-extension" )
      ;

    return plugin_loader().add_factory( fact );
  }
};

} // end namespace

#endif // PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_
