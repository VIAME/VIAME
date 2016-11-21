/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include <vital/vital_config.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <sprokit/pipeline/types.h>

#include <functional>
#include <memory>

namespace sprokit {

// returns: process_t - shared_ptr<process>
typedef std::function< process_t( kwiver::vital::config_block_sptr const& config ) > process_factory_func_t;

// Uses native sprokit::create_process<process-class> from process_registry.h:168
// This will also work for creating clusters

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
  return boost::make_shared<T>(conf);
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
class process_factory
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
                   const std::string& itype
                   process_factory_func_t factory )
    : m_factory( factory )
  {
    this->add_attribute( CONCRETE_TYPE, type);
    this->add_attribute( INTERFACE_TYPE, itype );
  }

  virtual ~process_factory() VITAL_DEFAULT_DTOR

  virtual sprokit::process_t create_object(kwiver::vital::config_block_sptr const& config)
  {
    // Call sprokit factory function. Need to use this factory
    // function approach to handle clusters transparently.
    return m_factory( config );
  }

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
process_t create_process(process::type_t const& type,
                         process::name_t const& name,
                         kwiver::vital::config_block_sptr const& config = kwiver::vital::config_block::empty_config()) const;


/**
 * \brief Mark a process as loaded.
 *
 * \param module The process to mark as loaded.
 */
SPROKIT_PIPELINE_EXPORT
void mark_process_as_loaded(module_t const& module);


/**
 * \brief Query if a process has already been loaded.
 *
 * \param module The process to query.
 *
 * \returns True if the process has already been loaded, false otherwise.
 */
SPROKIT_PIPELINE_EXPORT
bool is_process_loaded(module_t const& module) const;


//
// Convenience macro for adding processes
//
#define ADD_PROCESS( proc_type )                                        \
  add_factory( new sprokit::process_factory( typeid( proc_type ).name(), \
                                             typeid( sprokit::process ).name(), \
                                             sprokit::create_new_process< proc_type > ) )

} // end namespace


// ------------------------------------------------------------------
#if 0 //+ for experimentation
// USAGE
// in registration.cxx for processes,


extern "C"
SPROKIT_PROCESSES_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_manager& vpm )
{
  static const auto module_name = kwiver::vital::plugin_manager::module_t( "example_processes" );

  if ( sprokit::is_process_module_loaded( module_name ) )
  {
    return;
  }

  // This could be wrapped in a larger macro/template
  kwiver::vital::plugin_factory_handle_t fact = vpm.ADD_PROCESS( any_source_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "any_source" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which creates arbitrary data" );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

  sprokit::mark_process_moduleas_loaded( module_name );
}

#endif
