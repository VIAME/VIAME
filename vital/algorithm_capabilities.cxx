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

#include "algorithm_capabilities.h"

#include <map>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
class algorithm_capabilities::priv
{
public:

  std::map< std::string, bool > m_capabilities;
};


// ==================================================================
algorithm_capabilities
::algorithm_capabilities()
  : d( new algorithm_capabilities::priv )
{
}


algorithm_capabilities
::algorithm_capabilities( algorithm_capabilities const& other )
  : d( new algorithm_capabilities::priv(*other.d) ) // copy private implementation
{
}


algorithm_capabilities
::~algorithm_capabilities()
{
}


// ------------------------------------------------------------------
algorithm_capabilities&
algorithm_capabilities
::operator=( algorithm_capabilities const& other )
{
  if ( this != &other)
  {
    this->d.reset( new algorithm_capabilities::priv( *other.d ) ); // copy private implementation
  }

  return *this;
}


// ------------------------------------------------------------------
bool
algorithm_capabilities
::has_capability( capability_name_t const& name ) const
{
  return ( d->m_capabilities.count( name ) > 0 );
}


// ------------------------------------------------------------------
algorithm_capabilities::capability_list_t
algorithm_capabilities
:: capability_list() const
{
  algorithm_capabilities::capability_list_t list;

  for (auto ix = d->m_capabilities.begin(); ix != d->m_capabilities.end(); ++ix )
  {
    list.push_back( ix->first );
  }

  return list;
}


// ------------------------------------------------------------------
bool
algorithm_capabilities
::capability( capability_name_t const& name ) const
{
  if ( ! has_capability( name ) )
  {
    return false;
  }

  return d->m_capabilities[name];
}


// ------------------------------------------------------------------
void
algorithm_capabilities
::set_capability( capability_name_t const& name, bool val )
{
  d->m_capabilities[name] = val;
}

} } // end namespace
