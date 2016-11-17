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

// ------------------------------------------------------------------
// sample code for process factory and cluster factory

/*


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
 * This is to help reduce the amount of code needed in registration functions.
 *
 * \param conf The configuration to pass to the \ref process.
 *
 * \returns The new process.
 */
template <typename T>
process_t
create_new_process(kwiver::vital::config_block_sptr const& conf)
{
  return std::make_shared<T>(conf);
}


// ----------------------------------------------------------------
/**
 * @brief Factory class for sprokit processes
 *
 * This class represents a factory class for sprokit processes.  This
 * specialized factory creates a specific process and returns a shared
 * pointer to the base class to support polymorphic behaviour. It also
 * requires a single argument to the factory method.
 *
 * \tparam C Concrete process class type.
 */
template< class C >
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
   * @param itype Type name of interface type.
   * @param factory The Factory function
   */
  process_factory( process_factory_func_t factory )
    : m_factory( factory )
  {
    // Set concrete type of factory
    this->add_attribute( CONCRETE_TYPE, typeid( C ).name() );
    this->add_attribute( INTERFACE_TYPE, typeid( sprokit::process ).name() );
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


//
// Convenience macro for adding processes
//
#define ADD_PROCESS( proc_type ) \
  add_factory( new sprokit::process_factory< proc_type >( sprokit::create_new_process< proc_type > ) )
} // end namespace


// ------------------------------------------------------------------
#if 01 //+ for experimentation
// USAGE
// in registration.cxx for processes,


extern "C"
SPROKIT_PROCESSES_EXAMPLES_EXPORT
void
register_factories( kwiver::vital::plugin_manager& vpm )
{
  //+ The module name is just a string. Do we need a specific semantic type for this? Maybe so.
  // The process registry is the process specific layer over the plugin manager. It provides the
  // legacy API for processes (e.g. create_process()
  static process_registry::module_t const module_name = process_registry::module_t("example_processes");

  if ( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // This could be wrapped in a larger macro/template
  kwiver::vital::plugin_factory_handle_t fact = pm->ADD_PROCESS( any_source_process );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, "any_source");
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, "A process which creates arbitrary data");
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" );

/*
  registry->register_process("any_source", "A process which creates arbitrary data", create_process<any_source_process>);
  registry->register_process("const", "A process with the const flag", create_process<const_process>);
  registry->register_process("const_number", "Outputs a constant number", create_process<const_number_process>);
  registry->register_process("data_dependent", "A process with a data dependent type", create_process<data_dependent_process>);
*/

  vpm.mark_module_as_loaded( module_name );
}

#endif
