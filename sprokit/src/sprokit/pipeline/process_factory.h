/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * @file   process_factory.h
 * @brief  Interface to sprokit process factory
 */

#ifndef SPROKIT_PIPELINE_PROCESS_FACTORY_H
#define SPROKIT_PIPELINE_PROCESS_FACTORY_H

#include "pipeline-config.h"

#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sprokit/pipeline/process.h>

#include <functional>
#include <memory>

#ifdef SPROKIT_ENABLE_PYTHON
  #include <pybind11/pybind11.h>
#endif

namespace sprokit {

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
 * @brief Factory class for sprokit processes
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
   * @brief CTOR for factory object
   *
   * This CTOR also takes a factory function so it can support
   * creating processes and clusters.
   *
   * @param type Type name of the process
   * @param itype Type name of interface type.
   * @param factory The Factory function
   */
  process_factory( const std::string& type,
                   const std::string& itype,
                   process_factory_func_t factory );

  virtual ~process_factory();

  virtual sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config);

private:
  process_factory_func_t m_factory;
};


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
 * @brief Get list of all processes.
 *
 * @return List of all process implementation factories.
 */
SPROKIT_PIPELINE_EXPORT
kwiver::vital::plugin_factory_vector_t const& get_process_list();

//
// Convenience macro for adding processes
//
#define ADD_PROCESS( proc_type )                          \
  add_factory( new sprokit::process_factory( typeid( proc_type ).name(), \
                                             typeid( sprokit::process ).name(), \
                                             sprokit::create_new_process< proc_type > ) )

#ifdef SPROKIT_ENABLE_PYTHON
typedef std::function< pybind11::object( kwiver::vital::config_block_sptr const& config ) > py_process_factory_func_t;

class SPROKIT_PIPELINE_EXPORT python_process_factory
: public kwiver::vital::plugin_factory
{
  /**
   * @brief CTOR for factory object
   *
   * This CTOR is designed to work in conjunction with pybind11
   *
   * @param type Type name of the process
   * @param itype Type name of interface type.
   * @param factory The Factory function
   */
  public:

  python_process_factory( const std::string& type,
                          const std::string& itype,
                          py_process_factory_func_t factory );

  virtual ~python_process_factory();

  virtual sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config);

private:
  py_process_factory_func_t m_factory;
};

SPROKIT_PIPELINE_EXPORT
sprokit::process_t create_py_process(const sprokit::process::type_t&        type,
                                     const sprokit::process::name_t&        name,
                                     const kwiver::vital::config_block_sptr config = kwiver::vital::config_block::empty_config() );
#endif

} // end namespace

#endif /* SPROKIT_PIPELINE_PROCESS_FACTORY_H */
