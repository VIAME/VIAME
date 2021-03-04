// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file   process_factory.h
 * \brief  Interface to sprokit process factory
 */

#ifndef SPROKIT_PIPELINE_PROCESS_FACTORY_H
#define SPROKIT_PIPELINE_PROCESS_FACTORY_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_registrar.h>

#include <sprokit/pipeline/process.h>

#include <functional>
#include <memory>

namespace sprokit {

class cluster_info;
using cluster_info_t =  std::shared_ptr<cluster_info>;

// returns: process_t - shared_ptr<process>
typedef std::function< process_t( kwiver::vital::config_block_sptr const& config ) > process_factory_func_t;

  /**
 * \brief A template function to create a process.
 *
 * This function is the factory function for processes. This extra
 * level of factory is needed so that the process_factory class can
 * transparently support creating clusters in the same way as
 * processes.
 *
 * \param conf The configuration to pass to the \ref process.
 *
 * \returns The new process.
 */
template <typename T>
process_t
create_new_process(kwiver::vital::config_block_sptr const& conf)
{
  // Note shared pointer
  return std::make_shared<T>(conf);
}

// ----------------------------------------------------------------
/**
 * \brief Factory class for sprokit processes
 *
 * This class represents a factory class for sprokit processes and
 * clusters.  This specialized factory creates a specific process and
 * returns a shared pointer to the base class to support polymorphic
 * behaviour. It also requires a single argument to the factory
 * method. This works as a cluster factory because a cluster looks
 * like a process once it is created.
 *
 * \tparam C Concrete process class type.
 */
class SPROKIT_PIPELINE_EXPORT process_factory
: public kwiver::vital::plugin_factory
{
public:
  /**
   * \brief constructor for factory object
   *
   * This constructor also takes a factory function so it can support
   * creating processes and clusters.
   *
   * \param type Type name of the process
   * \param itype Type name of interface type.
   */
  process_factory( const std::string& type,
                   const std::string& itype );

  virtual ~process_factory() = default;

  virtual sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config) = 0;

  void copy_attributes( sprokit::process_t proc );
};

// ----------------------------------------------------------------------------
/**
 * \brief factory for CPP processes.
 *
 * This class represents the factory for a CPP process.
 */
class SPROKIT_PIPELINE_EXPORT cpp_process_factory
: public process_factory
{
public:
  /**
   * \brief CTOR for factory object
   *
   * This CTOR also takes a factory function so it can support
   * creating processes and clusters.
   *
   * \param type Type name of the process class.
   * \param itype Type name of interface type (usually "process").
   * \param factory The Factory function
   */
  cpp_process_factory( const std::string& type,
                       const std::string& itype,
                       process_factory_func_t factory );

  virtual ~cpp_process_factory() = default;

  sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config) override;

private:
  process_factory_func_t m_factory;
};

// ----------------------------------------------------------------------------
/**
 * \brief Factory class for clusters
 *
 * This class represents a factory for clusters of processes.
 * The description for the cluster is in the cluster info element.
 */
class SPROKIT_PIPELINE_EXPORT cluster_process_factory
: public process_factory
{
public:
  cluster_process_factory( cluster_info_t info );

  virtual ~cluster_process_factory() = default;

  sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config) override;

  cluster_info_t m_cluster_info;
};

// ----------------------------------------------------------------------------
/**
 * \brief Create process of a specific type.
 *
 * \throws no_such_process_type_exception Thrown if the type is not known.
 *
 * \param type The type of \ref process to create.
 * \param name The name of the \ref process to create.
 * \param config The configuration to pass the \ref process.
 *
 * \returns A new process of type \p type.
 */
SPROKIT_PIPELINE_EXPORT
sprokit::process_t create_process(const sprokit::process::type_t&        type,
                                  const sprokit::process::name_t&        name,
                                  const kwiver::vital::config_block_sptr config = kwiver::vital::config_block::empty_config() );

/**
 * \brief Mark a process as loaded.
 *
 * \param vpl The loader object that is managing the list of loadable modules.
 * \param module The process to mark as loaded.
 */
SPROKIT_PIPELINE_EXPORT
  void mark_process_module_as_loaded( kwiver::vital::plugin_loader& vpl,
                                      const module_t& module );

/**
 * \brief Query if a process has already been loaded.
 *
 * \param vpl The loader object that is managing the list of loadable modules.
 * \param module The process to query.
 *
 * \returns True if the process has already been loaded, false otherwise.
 */
SPROKIT_PIPELINE_EXPORT
  bool is_process_module_loaded( kwiver::vital::plugin_loader& vpl,
                                 module_t const& module );

/**
 * \brief Get list of all processes.
 *
 * \return List of all process implementation factories.
 */
SPROKIT_PIPELINE_EXPORT
kwiver::vital::plugin_factory_vector_t const& get_process_list();

//
// Convenience macro for adding processes
//
#define ADD_PROCESS( proc_type )                                        \
  add_factory( new sprokit::cpp_process_factory( typeid( proc_type ).name(), \
                                                 typeid( sprokit::process ).name(), \
                                                 sprokit::create_new_process< proc_type > ) )

// ============================================================================
/// Derived class to register processes
/**
 * This derived class contains the specific procedure for registering
 * processes with the plugin loader.
 */
class process_registrar
  : public kwiver::plugin_registrar
{
public:
  enum option {
    none = 0,
    no_test = 1
  };

  process_registrar( kwiver::vital::plugin_loader& vpl,
                       const std::string& mod_name_ )
    : plugin_registrar( vpl, mod_name_ )
  {
  }

  // Use forced naming convention for processes
  bool is_module_loaded() override
  {
    return plugin_loader().is_module_loaded( "process." + module_name() );
  }

  void mark_module_as_loaded() override
  {
    plugin_loader().mark_module_as_loaded( "process." + module_name() );
  }

  // ----------------------------------------------------------------------------
  /// Register a process plugin.
  /**
   * A process of the specified type is registered with the plugin
   * manager.
   *
   * \tparam tool_t Type of the process being registered.
   *
   * \return The plugin loader reference is returned.
   */
  template <typename process_t>
  kwiver::vital::plugin_factory_handle_t
  register_process( option opt = none )
  {
    using kvpf = kwiver::vital::plugin_factory;

    kwiver::vital::plugin_factory* fact =  new sprokit::cpp_process_factory(
      typeid( process_t ).name(),
      typeid( sprokit::process ).name(),
      sprokit::create_new_process< process_t > );

    fact->add_attribute( kvpf::PLUGIN_NAME,      process_t::_plugin_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,  process_t::_plugin_description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,  this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION, this->organization() )
      ;

    if (opt == no_test)
    {
      fact->add_attribute( "no-test", "introspect" ); // do not include in introspection test
    }

    return plugin_loader().add_factory( fact );
  }
};

} // end namespace

#endif /* SPROKIT_PIPELINE_PROCESS_FACTORY_H */
