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

#include "scheduler_factory.h"
#include "scheduler_registry_exception.h"

#include <vital/logger/logger.h>

namespace sprokit {

scheduler::type_t const scheduler_factory::default_type = scheduler::type_t("thread_per_process");


// ------------------------------------------------------------------
sprokit::scheduler_t create_scheduler( const sprokit::scheduler::type_t&      name,
                                       const sprokit::pipeline_t&             pipe,
                                       const kwiver::vital::config_block_sptr config )
{
  if ( ! config )
  {
    throw null_scheduler_registry_config_exception();
  }

  if (!pipe)
  {
    throw null_scheduler_registry_pipeline_exception();
  }

#ifdef SPROKIT_ENABLE_PYTHON
  // If python is enabled, we need to check for a python factory first
  try
  {
    auto proc = create_py_scheduler(name, pipe, config);
    return proc;
  }
  catch ( const std::exception &e )
  {
    auto logger = kwiver::vital::get_logger( "sprokit.object_factory" );
    LOG_DEBUG( logger, "No python scheduler found, trying C++");
  }
#endif

  typedef kwiver::vital::implementation_factory_by_name< sprokit::scheduler > instrumentation_factory;
  instrumentation_factory ifact;

  kwiver::vital::plugin_factory_handle_t a_fact;
  try
  {
    a_fact = ifact.find_factory( name );
  }
  catch ( kwiver::vital::plugin_factory_not_found& e )
  {
    auto logger = kwiver::vital::get_logger( "sprokit.scheduler_factory" );
    LOG_DEBUG( logger, "Plugin factory not found: " << e.what() );

    throw no_such_scheduler_type_exception( name );
  }

  sprokit::scheduler_factory* pf = dynamic_cast< sprokit::scheduler_factory* > ( a_fact.get() );
  if (0 == pf)
  {
    // wrong type of factory returned
    throw no_such_scheduler_type_exception( name );
  }

  return pf->create_object( pipe, config );
}


// ------------------------------------------------------------------
void
mark_scheduler_module_as_loaded( kwiver::vital::plugin_loader& vpl,
                                 module_t const& module )
{
  module_t mod = "scheduler.";
  mod += module;

  vpl.mark_module_as_loaded( mod );
}


// ------------------------------------------------------------------
bool
is_scheduler_module_loaded( kwiver::vital::plugin_loader& vpl,
                            module_t const& module )
{
  module_t mod = "scheduler.";
  mod += module;

  return vpl.is_module_loaded( mod );
}

// ------------------------------------------------------------------
// TODO: Most of this code is duplicate, and can be rewritten to use templates
#ifdef SPROKIT_ENABLE_PYTHON
python_scheduler_factory::
python_scheduler_factory( const std::string& type,
                          const std::string& itype,
                          py_scheduler_factory_func_t factory )
  : plugin_factory( itype )
  , m_factory( factory )
{
  this->add_attribute( CONCRETE_TYPE, type)
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid(* this ).name() )
    .add_attribute( PLUGIN_CATEGORY, "scheduler" );
}

python_scheduler_factory::
~python_scheduler_factory()
{ }

scheduler_t
python_scheduler_factory::
create_object(sprokit::pipeline_t const& pipe, kwiver::vital::config_block_sptr const& config)
{
  // Call sprokit factory function. Need to use this factory
  // function approach to handle clusters transparently.
  pybind11::object obj = m_factory(pipe, config);
  obj.inc_ref();
  sprokit::scheduler_t schd_ptr = obj.cast<sprokit::scheduler_t>();
  return schd_ptr;
}


// ------------------------------------------------------------------
scheduler_t
create_py_scheduler( const sprokit::scheduler::type_t&      name,
                     const sprokit::pipeline_t&             pipe,
                     const kwiver::vital::config_block_sptr config )
{
  // First see if there's a C++ scheduler with the name
  // If that fails, try python instead
  typedef kwiver::vital::implementation_factory_by_name< pybind11::object > instrumentation_factory;
  instrumentation_factory ifact;

  kwiver::vital::plugin_factory_handle_t a_fact;
  try
  {
    a_fact = ifact.find_factory( name );
  }
  catch ( kwiver::vital::plugin_factory_not_found& e )
  {
    auto logger = kwiver::vital::get_logger( "python_scheduler_factory" );
    LOG_DEBUG( logger, "Plugin factory not found: " << e.what() );

    throw sprokit::no_such_scheduler_type_exception( name );
  }

  sprokit::python_scheduler_factory* pf = dynamic_cast< sprokit::python_scheduler_factory* > ( a_fact.get() );
  if (0 == pf)
  {
    // wrong type of factory returned
    throw sprokit::no_such_scheduler_type_exception( name );
  }

  return pf->create_object( pipe, config );
}

#endif


} // end namespace
