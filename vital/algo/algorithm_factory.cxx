/*ckwg +29
 * Copyright 2016, 2020 by Kitware, Inc.
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
