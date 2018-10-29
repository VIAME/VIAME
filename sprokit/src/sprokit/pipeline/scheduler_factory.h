/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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
