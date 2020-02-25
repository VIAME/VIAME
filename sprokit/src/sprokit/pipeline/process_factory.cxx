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

#include "process_factory.h"
#include "process_registry_exception.h"

#include <vital/util/tokenize.h>
#include <vital/logger/logger.h>

#include <algorithm>

namespace sprokit {

// ------------------------------------------------------------------
process_factory::
process_factory( const std::string& type,
                 const std::string& itype )
  : plugin_factory( itype )
{
  this->add_attribute( CONCRETE_TYPE, type)
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid(* this ).name() )
    .add_attribute( PLUGIN_CATEGORY, PROCESS_CATEGORY );
}


// ----------------------------------------------------------------------------
void
process_factory::
copy_attributes( sprokit::process_t proc )
{
  // Add any properties from the factory attributes
  std::string props;
  if ( get_attribute( kwiver::vital::plugin_factory::PLUGIN_PROCESS_PROPERTIES, props ) )
  {
    // split props by " " or "," then add to process props.
    std::vector< std::string > fact_props;
    kwiver::vital::tokenize( props, fact_props, ", ", kwiver::vital::TokenizeTrimEmpty );
    for ( std::string a_prop : fact_props )
    {
      proc->add_property( a_prop );
    }
  }
}


// ============================================================================
cpp_process_factory::
cpp_process_factory( const std::string& type,
                 const std::string& itype,
                 process_factory_func_t factory )
  : process_factory( type, itype )
  , m_factory( factory )
{
  this->add_attribute( CONCRETE_TYPE, type)
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid(* this ).name() )
    .add_attribute( PLUGIN_CATEGORY, PROCESS_CATEGORY );
}


// ------------------------------------------------------------------
sprokit::process_t
cpp_process_factory::
create_object(kwiver::vital::config_block_sptr const& config)
{
  // Call sprokit factory function. Need to use this factory
  // function approach to handle clusters transparently.
  sprokit::process_t proc = m_factory( config );

  // Copy attributes from factory to process.
  copy_attributes( proc );

  return proc;
}


// ------------------------------------------------------------------
sprokit::process_t
create_process( const sprokit::process::type_t&         type,
                const sprokit::process::name_t&         name,
                const kwiver::vital::config_block_sptr  config )
{
  if ( ! config )
  {
    VITAL_THROW( null_process_registry_config_exception );
  }

  typedef kwiver::vital::implementation_factory_by_name< sprokit::process > proc_factory;
  proc_factory ifact;

  kwiver::vital::plugin_factory_handle_t a_fact;
  try
  {
    a_fact = ifact.find_factory( type );
  }
  catch ( kwiver::vital::plugin_factory_not_found& e )
  {
    auto logger = kwiver::vital::get_logger( "sprokit.process_factory" );
    LOG_DEBUG( logger, "Plugin factory not found: " << e.what() );

    VITAL_THROW( no_such_process_type_exception, type );
  }

  // Add these entries to the new process config so it will know how it is instantiated.
  config->set_value( process::config_type, kwiver::vital::config_block_value_t( type ) );
  config->set_value( process::config_name, kwiver::vital::config_block_value_t( name ) );

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( a_fact.get() );
  if (0 == pf)
  {
    // Wrong type of factory returned.
    VITAL_THROW( no_such_process_type_exception, type );
  }

  sprokit::process_t proc;
  try
  {
    proc = pf->create_object( config );
  }
  catch ( const std::exception &e )
  {
    auto logger = kwiver::vital::get_logger( "sprokit.process_factory" );
    LOG_ERROR( logger, "Exception from creating process: " << e.what() );
    throw;
  }

  return proc;
}


// ------------------------------------------------------------------
void
mark_process_module_as_loaded( kwiver::vital::plugin_loader& vpl,
                               module_t const& module )
{
  module_t mod = "process.";
  mod += module;

  vpl.mark_module_as_loaded( mod );
}


// ------------------------------------------------------------------
bool
is_process_module_loaded( kwiver::vital::plugin_loader& vpl,
                          module_t const& module )
{
  module_t mod = "process.";
  mod += module;

  return vpl.is_module_loaded( mod );
}


// ------------------------------------------------------------------
kwiver::vital::plugin_factory_vector_t const& get_process_list()
{
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  return vpm.get_factories<sprokit::process>();
}

} // end namespace
