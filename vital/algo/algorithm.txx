// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief implementation of algorithm/_def/_impl templated methods
 */

#ifndef VITAL_ALGO_ALGORITHM_TXX_
#define VITAL_ALGO_ALGORITHM_TXX_

#include "algorithm.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <typeinfo>

#include <vital/algo/algorithm_factory.h>
#include <vital/exceptions/algorithm.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Factory method to make an instance of this algorithm by impl_name
template < typename Self >
std::shared_ptr< Self >
algorithm_def< Self >
::create( const std::string& impl_name )
{
  return std::dynamic_pointer_cast< Self > ( create_algorithm( Self::static_type_name(), impl_name ) );
}

// ------------------------------------------------------------------
/// Return a vector of the impl_name of each registered implementation
template < typename Self >
std::vector< std::string >
algorithm_def< Self >
::registered_names()
{
  std::vector< std::string > names;

  auto fact_list = kwiver::vital::plugin_manager::instance().get_factories( Self::static_type_name() );

  for( auto fact : fact_list )
  {
    std::string attr_val;
    if (fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, attr_val ) )
    {
      names.push_back( attr_val );
    }
  }

  return names;
}

// ------------------------------------------------------------------
/// Helper function for properly getting a nested algorithm's configuration
template < typename Self >
void
algorithm_def< Self >
::get_nested_algo_configuration( std::string const&                name,
                                 kwiver::vital::config_block_sptr  config,
                                 base_sptr                         nested_algo )
{
  algorithm::get_nested_algo_configuration( Self::static_type_name(),
                                            name, config, nested_algo );
}

// ------------------------------------------------------------------
/// Helper macro for properly setting a nested algorithm's configuration
template < typename Self >
void
algorithm_def< Self >
::set_nested_algo_configuration( std::string const&                name,
                                 kwiver::vital::config_block_sptr  config,
                                 base_sptr&                        nested_algo )
{
  algorithm_sptr base_nested_algo =
    std::static_pointer_cast< algorithm > ( nested_algo );
  algorithm::set_nested_algo_configuration( Self::static_type_name(),
                                            name, config, base_nested_algo );

  nested_algo = std::dynamic_pointer_cast< Self > ( base_nested_algo );
}

// ------------------------------------------------------------------
/// Helper macro for checking that basic nested algorithm configuration is valid
template < typename Self >
bool
algorithm_def< Self >
::check_nested_algo_configuration( std::string const&                name,
                                   kwiver::vital::config_block_sptr  config )
{
  return algorithm::check_nested_algo_configuration( Self::static_type_name(),
                                                     name, config );
}

}
}     // end namespace

/// \cond DoxygenSuppress
#define INSTANTIATE_ALGORITHM_DEF( T ) \
  template class kwiver::vital::algorithm_def< T >;
/// \endcond
#endif // VITAL_ALGO_ALGORITHM_TXX_
