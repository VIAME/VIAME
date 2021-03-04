// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "scheduler_factory.h"
#include "scheduler_registry_exception.h"

#include <vital/logger/logger.h>

namespace sprokit {

scheduler::type_t const scheduler_factory::default_type = scheduler::type_t("thread_per_process");

// ----------------------------------------------------------------------------
scheduler_factory::
scheduler_factory( const std::string&       type,
                   const std::string&       itype )
    : plugin_factory( itype )
  {
    this->add_attribute( CONCRETE_TYPE, type)
      .add_attribute( PLUGIN_FACTORY_TYPE, typeid( *this ).name() )
      .add_attribute( PLUGIN_CATEGORY, "scheduler" );
  }

// ----------------------------------------------------------------------------
cpp_scheduler_factory::
cpp_scheduler_factory( const std::string&       type,
                       const std::string&       itype,
                       scheduler_factory_func_t factory )
  : scheduler_factory( type, itype )
  , m_factory( factory )
{
  this->add_attribute( CONCRETE_TYPE, type)
    .add_attribute( PLUGIN_FACTORY_TYPE, typeid( *this ).name() )
    .add_attribute( PLUGIN_CATEGORY, "scheduler" );
}

// ----------------------------------------------------------------------------
sprokit::scheduler_t
cpp_scheduler_factory::
create_object( pipeline_t const& pipe,
               kwiver::vital::config_block_sptr const& config )
{
  // Call sprokit factory function.
  return m_factory( pipe, config );
}

// ------------------------------------------------------------------
sprokit::scheduler_t create_scheduler( const sprokit::scheduler::type_t&      name,
                                       const sprokit::pipeline_t&             pipe,
                                       const kwiver::vital::config_block_sptr config )
{
  if ( ! config )
  {
    VITAL_THROW( null_scheduler_registry_config_exception );
  }

  if (!pipe)
  {
    VITAL_THROW( null_scheduler_registry_pipeline_exception );
  }

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

    VITAL_THROW( no_such_scheduler_type_exception, name );
  }

  sprokit::scheduler_factory* pf = dynamic_cast< sprokit::scheduler_factory* > ( a_fact.get() );
  if (0 == pf)
  {
    // wrong type of factory returned
    VITAL_THROW( no_such_scheduler_type_exception, name );
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

} // end namespace
