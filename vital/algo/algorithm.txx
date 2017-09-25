/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
