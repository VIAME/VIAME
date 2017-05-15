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

/**
 * @file   provider_dereferencer.cxx
 * @brief  Implementation for provider_dereferencer class.
 */

#include "provider_dereferencer.h"

#include <memory>

namespace sprokit {

// ------------------------------------------------------------------
provider_dereferencer
::provider_dereferencer()
  : m_providers()
{
  m_providers[bakery_base::provider_system] = std::make_shared< system_provider > ();
  m_providers[bakery_base::provider_environment] = std::make_shared< environment_provider > ();
}


// ------------------------------------------------------------------
provider_dereferencer
::provider_dereferencer( kwiver::vital::config_block_sptr const conf )
  : m_providers()
{
  m_providers[bakery_base::provider_config] = std::make_shared< config_provider > ( conf );
}


// ------------------------------------------------------------------
provider_dereferencer
::~provider_dereferencer()
{
}


// ------------------------------------------------------------------
bakery_base::config_reference_t
provider_dereferencer
::operator()( kwiver::vital::config_block_value_t const& value ) const
{
  return value;
}


// ------------------------------------------------------------------
bakery_base::config_reference_t
provider_dereferencer
::operator()( bakery_base::provider_request_t const& request ) const
{
  config_provider_t const& provider_name = request.first;
  provider_map_t::const_iterator const i = m_providers.find( provider_name );

  if ( i == m_providers.end() )
  {
    return request;
  }

  provider_t const& provider = i->second;
  kwiver::vital::config_block_value_t const& value = request.second;

  return ( *provider )( value );
}

} // end namespace sprokit
