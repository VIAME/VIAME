// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_
#define PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_

#include <viame/pipeline_framework/adapters/kwiver_adapter_export.h>

#include <viame/pipeline_framework/pipeline.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/logger/logger.h>
#include <viame/algorithm_framework/plugin/plugin_info.h>
#include <viame/algorithm_framework/plugin/plugin_registrar.h>
#include <viame/algorithm_framework/vital_config.h>

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
  /// Interface name for plugin factory
  static std::string interface_name() { return "embedded_pipeline_extension"; }

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
  virtual void pre_setup( [[maybe_unused]] context& ctxt ) { };

  /**
   * @brief pipeline post-setup hook
   *
   * This method is called after the pipeline is setup. A context
   * object is supplied to this hook so it can query its running environment.
   *
   * @param ctxt The calling context.
   */
  virtual void post_setup( [[maybe_unused]] context& ctxt ) { };

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
  virtual void end_of_output( [[maybe_unused]] context& ctxt ) { };

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
  virtual void configure( [[maybe_unused]] kwiver::vital::config_block_sptr const conf );

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
/// Factory class for embedded pipeline extensions
class KWIVER_ADAPTER_EXPORT epx_factory
  : public kwiver::vital::plugin_factory
{
public:
  epx_factory( const std::string& type,
               const std::string& itype,
               const std::string& concrete_type )
    : plugin_factory( itype )
  {
    this->add_attribute( PLUGIN_NAME, type );
    this->add_attribute( CONCRETE_TYPE, concrete_type );
  }

  virtual ~epx_factory() = default;

  virtual embedded_pipeline_extension_sptr create_object() = 0;

  // Implement pure virtual methods from plugin_factory base class
  kwiver::vital::pluggable_sptr from_config( kwiver::vital::config_block_sptr const ) const override
  {
    return nullptr;
  }

  void get_default_config( kwiver::vital::config_block& ) const override
  {
  }
};

// ============================================================================
/// Templated factory for concrete EPX types
template< typename T >
class epx_factory_impl
  : public epx_factory
{
public:
  epx_factory_impl( const std::string& type,
                    const std::string& itype,
                    const std::string& concrete_type )
    : epx_factory( type, itype, concrete_type )
  {
  }

  virtual embedded_pipeline_extension_sptr create_object() override
  {
    return std::make_shared< T >();
  }
};

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
  embedded_pipeline_extension_registrar( kwiver::vital::plugin_loader& p_vpl,
                    const std::string& p_module_name )
    : plugin_registrar( p_vpl, p_module_name )
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

    kwiver::vital::plugin_factory* fact = new epx_factory_impl< epx_t >(
      epx_t::_plugin_name,
      typeid( kwiver::embedded_pipeline_extension ).name(),
      typeid( epx_t ).name() );

    fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  epx_t::_plugin_description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,  this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION, this->organization() )
      .add_attribute( kwiver::vital::plugin_factory::PLUGIN_CATEGORY, "embedded-pipeline-extension" )
      ;

    return plugin_loader().add_factory( fact );
  }
};

} // end namespace

#endif // PROCESS_ADAPTERS_EMBEDDED_PIPELINE_EXTENSION_H_
