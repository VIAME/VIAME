// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file   process_factory.h
 * \brief  Interface to sprokit process factory
 */

#ifndef SPROKIT_PIPELINE_PROCESS_FACTORY_H
#define SPROKIT_PIPELINE_PROCESS_FACTORY_H

#include <viame/pipeline_framework/sprokit_pipeline_export.h>

#include <viame/algorithm_framework/vital_config.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/plugin/plugin_registrar.h>

#include <viame/pipeline_framework/process.h>

#include <functional>
#include <map>
#include <memory>

namespace viame::pipeline {


// returns: process_t - shared_ptr<process>
typedef std::function< process_t( viame::config_block_sptr const& config ) > process_factory_func_t;

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
create_new_process(viame::config_block_sptr const& conf)
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
: public viame::plugin_factory
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

  virtual viame::pipeline::process_t create_object(viame::config_block_sptr const& config) = 0;

  void copy_attributes( viame::pipeline::process_t proc );

  // Implement pure virtual methods from plugin_factory base class
  // Sprokit processes use their own configuration mechanism, so these are stubs
  viame::pluggable_sptr from_config( [[maybe_unused]] viame::config_block_sptr const cb ) const override
  {
    // Sprokit processes are not pluggable in the same way as vital algorithms
    return nullptr;
  }

  void get_default_config( [[maybe_unused]] viame::config_block& cb ) const override
  {
    // Sprokit processes configure themselves differently
  }
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

  viame::pipeline::process_t create_object(viame::config_block_sptr const& config) override;

private:
  process_factory_func_t m_factory;
};

// ----------------------------------------------------------------------------
/**
 * \brief Give a process type a second name.
 *
 * A process that is renamed keeps working under its old type, and the
 * baseline diff can tell a rename from a removal, which is the whole point:
 * `registry.json` lists names, so a process that changed its own is
 * indistinguishable from one that was deleted and one that was added.
 *
 * Algorithms have had this since the imports (`VIAME_REGISTER_ALIAS`), where
 * it is a second registration of the same class. Processes could not do the
 * same -- `register_process` takes the name from `process_t::_plugin_name`,
 * a static on the class, so there was nowhere to put the second one. This is
 * a table instead, which is also what `lite-build-system.md` 4 asks for.
 *
 * Resolution happens in `create_process`, so it covers every route into a
 * process: the pipe bakery, the embedded pipeline, and anything else that
 * names a type.
 *
 * \param alias The old type name.
 * \param target The type it now resolves to.
 */
SPROKIT_PIPELINE_EXPORT
void add_process_alias( viame::pipeline::process::type_t const& alias,
                        viame::pipeline::process::type_t const& target );

/**
 * \brief Every process type alias, old name to new.
 */
SPROKIT_PIPELINE_EXPORT
std::map< viame::pipeline::process::type_t, viame::pipeline::process::type_t >
process_aliases();

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
viame::pipeline::process_t create_process(const viame::pipeline::process::type_t&        type,
                                  const viame::pipeline::process::name_t&        name,
                                  const viame::config_block_sptr config = viame::config_block::empty_config() );

/**
 * \brief Mark a process as loaded.
 *
 * \param vpl The loader object that is managing the list of loadable modules.
 * \param module The process to mark as loaded.
 */
SPROKIT_PIPELINE_EXPORT
  void mark_process_module_as_loaded( viame::registry& vpl,
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
  bool is_process_module_loaded( viame::registry& vpl,
                                 module_t const& module );

/**
 * \brief Get list of all processes.
 *
 * \return List of all process implementation factories.
 */
SPROKIT_PIPELINE_EXPORT
viame::plugin_factory_vector_t const& get_process_list();

//
// Convenience macro for adding processes
//
#define ADD_PROCESS( proc_type )                                        \
  add_factory( new viame::pipeline::cpp_process_factory( typeid( proc_type ).name(), \
                                                 viame::pipeline::process::interface_name(), \
                                                 viame::pipeline::create_new_process< proc_type > ) )

/// Convenience macro to create a process factory (for use in tests)
#define MAKE_PROCESS_FACTORY( proc_type ) \
  new viame::pipeline::cpp_process_factory( typeid( proc_type ).name(), \
                                    viame::pipeline::process::interface_name(), \
                                    viame::pipeline::create_new_process< proc_type > )

// ============================================================================
/// Derived class to register processes
/**
 * This derived class contains the specific procedure for registering
 * processes with the registry.
 */
class process_registrar
  : public viame::plugin_registrar
{
public:
  enum option {
    none = 0,
    no_test = 1
  };

  process_registrar( viame::registry& vpl,
                       const std::string& mod_name_ )
    : plugin_registrar( vpl, mod_name_ )
  {
  }

  // Use forced naming convention for processes
  bool is_module_loaded() override
  {
    return registry().is_module_loaded( "process." + module_name() );
  }

  void mark_module_as_loaded() override
  {
    registry().mark_module_as_loaded( "process." + module_name() );
  }

  // ----------------------------------------------------------------------------
  /// Register a process plugin.
  /**
   * A process of the specified type is registered with the plugin
   * manager.
   *
   * \tparam tool_t Type of the process being registered.
   *
   * \return the registry reference is returned.
   */
  template <typename process_t>
  viame::plugin_factory_handle_t
  register_process( option opt = none )
  {
    using kvpf = viame::plugin_factory;

    viame::plugin_factory* fact =  new viame::pipeline::cpp_process_factory(
      typeid( process_t ).name(),
      viame::pipeline::process::interface_name(),
      viame::pipeline::create_new_process< process_t > );

    fact->add_attribute( kvpf::PLUGIN_NAME,      process_t::_plugin_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,  process_t::_plugin_description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,  this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION, this->organization() )
      ;

    if (opt == no_test)
    {
      fact->add_attribute( "no-test", "introspect" ); // do not include in introspection test
    }

    return registry().add_factory( fact );
  }
};

} // namespace viame::pipeline

#endif /* SPROKIT_PIPELINE_PROCESS_FACTORY_H */
