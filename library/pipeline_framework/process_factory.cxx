// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "process_factory.h"
#include "process_registry_exception.h"

#include <viame/algorithm_framework/util/tokenize.h>
#include <viame/algorithm_framework/logger/logger.h>

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
  this->add_attribute( PLUGIN_FACTORY_TYPE, typeid(* this ).name() )
    .add_attribute( PLUGIN_CATEGORY, PROCESS_CATEGORY );
}

// ------------------------------------------------------------------
sprokit::process_t
cpp_process_factory::
create_object(kwiver::vital::config_block_sptr const& config)
{
  sprokit::process_t proc = m_factory( config );

  // Copy attributes from factory to process.
  copy_attributes( proc );

  return proc;
}

// ============================================================================
namespace {

// Old type name to current one. A map rather than a member of the loader
// because the loader is kwiver's and the aliases are VIAME's; P8-T03's
// registry is where the two become one table.
std::map< sprokit::process::type_t, sprokit::process::type_t >&
process_alias_table()
{
  static std::map< sprokit::process::type_t, sprokit::process::type_t > table;
  return table;
}

} // namespace

// ------------------------------------------------------------------
void
add_process_alias( sprokit::process::type_t const& alias,
                   sprokit::process::type_t const& target )
{
  process_alias_table()[ alias ] = target;
}

// ------------------------------------------------------------------
std::map< sprokit::process::type_t, sprokit::process::type_t >
process_aliases()
{
  return process_alias_table();
}

// ============================================================================
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

  process::type_t resolved = type;

  kwiver::vital::plugin_factory_handle_t a_fact;
  try
  {
    a_fact = ifact.find_factory( resolved );
  }
  catch ( kwiver::vital::plugin_factory_not_found& e )
  {
    auto logger = kwiver::vital::get_logger( "sprokit.process_factory" );

    // A type nothing registers may be one that was renamed. The alias is
    // tried once, and only after the real name has failed, so a live type
    // never pays for the table.
    auto const& aliases = process_alias_table();
    auto const alias = aliases.find( type );

    if ( alias == aliases.end() )
    {
      LOG_DEBUG( logger, "Plugin factory not found: " << e.what() );

      VITAL_THROW( no_such_process_type_exception, type );
    }

    resolved = alias->second;

    LOG_DEBUG( logger, "Process type \"" << type << "\" is an alias for \""
                       << resolved << "\"" );

    try
    {
      a_fact = ifact.find_factory( resolved );
    }
    catch ( kwiver::vital::plugin_factory_not_found& inner )
    {
      // The alias names something that is not registered either, which is a
      // mistake in whoever added it rather than in the pipeline. Report the
      // name the caller used.
      LOG_ERROR( logger, "Process type \"" << type << "\" is an alias for \""
                         << resolved << "\", which is not registered: "
                         << inner.what() );

      VITAL_THROW( no_such_process_type_exception, type );
    }
  }

  // Add these entries to the new process config so it will know how it is instantiated.
  config->set_value( process::config_type, kwiver::vital::config_block_value_t( resolved ) );
  config->set_value( process::config_name, kwiver::vital::config_block_value_t( name ) );

  sprokit::process_factory* pf = dynamic_cast< sprokit::process_factory* > ( a_fact.get() );
  if (!pf)
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
