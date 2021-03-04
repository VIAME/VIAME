// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   scheduler_factory.h
 * @brief  Interface to sprokit scheduler factory
 */

#ifndef SPROKIT_PIPELINE_SCHEDULER_FACTORY_H
#define SPROKIT_PIPELINE_SCHEDULER_FACTORY_H

#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/scheduler.h>

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
//
#define ADD_SCHEDULER( type )                                           \
  add_factory( new sprokit::cpp_scheduler_factory( typeid( type ).name(), \
                                                   typeid( sprokit::scheduler ).name(), \
                                                   sprokit::create_new_scheduler< type > ) )

} // end namespace

#endif /* SPROKIT_PIPELINE_SCHEDULER_FACTORY_H */
