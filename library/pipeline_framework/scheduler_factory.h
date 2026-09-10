// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   scheduler_factory.h
 * @brief  Interface to sprokit scheduler factory
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_FACTORY_H
#define SPROKIT_PIPELINE_SCHEDULER_FACTORY_H

#include <viame/algorithm_framework/vital_config.h>
#include <viame/algorithm_framework/config/config_block.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/algorithm_framework/plugin/plugin_registrar.h>

#include <viame/pipeline_framework/scheduler.h>

#include <functional>
#include <memory>

namespace sprokit {

// returns: scheduler_t - shared_ptr<scheduler>
typedef std::function< scheduler_t( pipeline_t const& pipe,
        kwiver::vital::config_block_sptr const& config ) > scheduler_factory_func_t;

// ------------------------------------------------------------------
/**
 * \brief A template function to create a scheduler.
 *
 * This is to help reduce the amount of code needed in registration functions.
 *
 * \param conf The configuration to pass to the \ref scheduler.
 * \param pipe The \ref pipeline to pass the \ref scheduler.
 *
 * \return The new scheduler.
 */
template <typename T>
scheduler_t
create_new_scheduler( pipeline_t const& pipe,
                      kwiver::vital::config_block_sptr const& conf)
{
  return std::make_shared<T>(pipe, conf);
}

// ----------------------------------------------------------------
/**
 * @brief Factory class for sprokit schedulers
 *
 * This class represents a factory class for sprokit schedulers and
 * clusters.  This specialized factory creates a specific scheduler and
 * returns a shared pointer to the base class to support polymorphic
 * behaviour. It also requires a single argument to the factory
 * method.
 *
 * \tparam C Concrete scheduler class type.
 */
class SPROKIT_PIPELINE_EXPORT scheduler_factory
: public kwiver::vital::plugin_factory
{
public:
  static scheduler::type_t const default_type;

  /**
   * @brief CTOR for factory object
   *
   * This CTOR also takes a factory function so it can support
   * creating schedulers.
   *
   * @param type Type name of the scheduler
   * @param itype Type name of interface type.
   */
  scheduler_factory( const std::string&       type,
                     const std::string&       itype );

  virtual ~scheduler_factory() = default;

  virtual sprokit::scheduler_t create_object( pipeline_t const& pipe,
                                              kwiver::vital::config_block_sptr const& config ) = 0;

  // Implement pure virtual methods from plugin_factory base class
  // Sprokit schedulers use their own configuration mechanism, so these are stubs
  kwiver::vital::pluggable_sptr from_config( [[maybe_unused]] kwiver::vital::config_block_sptr const cb ) const override
  {
    // Sprokit schedulers are not pluggable in the same way as vital algorithms
    return nullptr;
  }

  void get_default_config( [[maybe_unused]] kwiver::vital::config_block& cb ) const override
  {
    // Sprokit schedulers configure themselves differently
  }
};

// ----------------------------------------------------------------------------
class SPROKIT_PIPELINE_EXPORT cpp_scheduler_factory
: public scheduler_factory
{
public:
  static scheduler::type_t const default_type;

  /**
   * @brief CTOR for factory object
   *
   * This CTOR also takes a factory function so it can support
   * creating schedulers.
   *
   * @param type Type name of the scheduler
   * @param itype Type name of interface type.
   * @param factory The Factory function
   */
  cpp_scheduler_factory( const std::string&       type,
                         const std::string&       itype,
                         scheduler_factory_func_t factory );

  virtual ~cpp_scheduler_factory() = default;

  virtual sprokit::scheduler_t create_object( pipeline_t const& pipe,
                                              kwiver::vital::config_block_sptr const& config );

private:
  scheduler_factory_func_t m_factory;
};

/**
 * \brief Create scheduler of a specific type.
 *
 * \throws no_such_scheduler_type_exception Thrown if the type is not known.
 *
 * \param nameXs! The name of the type of \ref scheduler to create.
 * \param pipe The \ref pipeline to pass the \ref scheduler.
 * \param config The configuration to pass the \ref scheduler.
 *
 * \returns A new scheduler of type \p type.
 */
SPROKIT_PIPELINE_EXPORT
sprokit::scheduler_t
create_scheduler( const sprokit::scheduler::type_t&      name,
                  const sprokit::pipeline_t&             pipe,
                  const kwiver::vital::config_block_sptr config = kwiver::vital::config_block::empty_config());

/**
 * \brief Mark a scheduler as loaded.
 *
 * \param vpl The loader object that is managing the list of loadable modules.
 * \param module The scheduler to mark as loaded.
 */
SPROKIT_PIPELINE_EXPORT
void mark_scheduler_module_as_loaded( kwiver::vital::plugin_loader& vpl,
                                      module_t const& module );

/**
 * \brief Query if a scheduler has already been loaded.
 *
 * \param vpl The loader object that is managing the list of loadable modules.
 * \param module The scheduler to query.
 *
 * \returns True if the scheduler has already been loaded, false otherwise.
 */
SPROKIT_PIPELINE_EXPORT
bool is_scheduler_module_loaded( kwiver::vital::plugin_loader& vpl,
                                 module_t const& module );

//
// Convenience macro for adding schedulers
// NOTE: This macro is deprecated. Use scheduler_registrar instead.
//
#define ADD_SCHEDULER( type )                                           \
  add_factory( new sprokit::cpp_scheduler_factory( typeid( type ).name(), \
                                                   sprokit::scheduler::interface_name(), \
                                                   sprokit::create_new_scheduler< type > ) )

/// Convenience macro to create a scheduler factory (for use in tests)
#define MAKE_SCHEDULER_FACTORY( type ) \
  new sprokit::cpp_scheduler_factory( typeid( type ).name(), \
                                      sprokit::scheduler::interface_name(), \
                                      sprokit::create_new_scheduler< type > )

// ============================================================================
/// Derived class to register schedulers
/**
 * This derived class contains the specific procedure for registering
 * schedulers with the plugin loader.
 */
class scheduler_registrar
  : public kwiver::plugin_registrar
{
public:
  scheduler_registrar( kwiver::vital::plugin_loader& vpl,
                       const std::string& mod_name_ )
    : plugin_registrar( vpl, mod_name_ )
  {
  }

  // Use forced naming convention for schedulers
  bool is_module_loaded() override
  {
    return plugin_loader().is_module_loaded( "scheduler." + module_name() );
  }

  void mark_module_as_loaded() override
  {
    plugin_loader().mark_module_as_loaded( "scheduler." + module_name() );
  }

  // ----------------------------------------------------------------------------
  /// Register a scheduler plugin.
  /**
   * A scheduler of the specified type is registered with the plugin
   * manager.
   *
   * \tparam scheduler_t Type of the scheduler being registered.
   * \param name Plugin name for the scheduler.
   * \param description Description of the scheduler.
   * \param version Version string (defaults to "1.0").
   *
   * \return The plugin loader reference is returned.
   */
  template <typename scheduler_t>
  kwiver::vital::plugin_factory_handle_t
  register_scheduler( const std::string& name,
                      const std::string& description,
                      const std::string& version = "1.0" )
  {
    using kvpf = kwiver::vital::plugin_factory;

    kwiver::vital::plugin_factory* fact = new sprokit::cpp_scheduler_factory(
      typeid( scheduler_t ).name(),
      sprokit::scheduler::interface_name(),
      sprokit::create_new_scheduler< scheduler_t > );

    fact->add_attribute( kvpf::PLUGIN_NAME,        name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,    description )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME,    this->module_name() )
      .add_attribute( kvpf::PLUGIN_ORGANIZATION,   this->organization() )
      .add_attribute( kvpf::PLUGIN_VERSION,        version )
      ;

    return plugin_loader().add_factory( fact );
  }
};

} // end namespace

#endif /* SPROKIT_PIPELINE_SCHEDULER_FACTORY_H */
