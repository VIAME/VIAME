// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "algorithm_factory.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
bool has_algorithm_impl_name(std::string const& type_name,
                             std::string const& impl_name)
{
    // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact_list = vpm.get_factories( type_name );

  // Find the one that provides the impl_name
  for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
  {
    std::string attr_val;
    if ( a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, attr_val )
         && ( attr_val == impl_name ) )
    {
      return true;
    }
  } // end foreach

  return false;
}

// ------------------------------------------------------------------
kwiver::vital::algorithm_sptr create_algorithm( std::string const& algo_name,
                                                std::string const& impl_name )
{
  // Get list of factories for the algo_name
  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  auto fact_list = vpm.get_factories( algo_name );

  // Find the one that provides the impl_name
  for( kwiver::vital::plugin_factory_handle_t a_fact : fact_list )
  {
    std::string attr_val;
    if ( a_fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, attr_val )
         && ( attr_val == impl_name ) )
    {
      kwiver::vital::algorithm_factory* pf = dynamic_cast< kwiver::vital::algorithm_factory* > ( a_fact.get() );
      if (0 == pf)
      {
        // Wrong type of factory returned.
        std::stringstream str;
        str << "Factory for algorithm name \"" << algo_name << "\" implementation \""
            << impl_name << "\" could not be converted to algorithm_factory type.";
        VITAL_THROW( kwiver::vital::plugin_factory_not_found, str.str() );
      }

      // created algorithm
      auto pa = pf->create_object();
      pa->set_impl_name( impl_name );
      return pa;
    }
  } // end foreach

  std::stringstream str;
  str << "Could not find factory for algorithm \"" << impl_name
      << "\" implementing \"" << algo_name << "\"";

  VITAL_THROW( kwiver::vital::plugin_factory_not_found, str.str() );
}

} } // end namesapce
