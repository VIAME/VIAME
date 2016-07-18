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

#include "plugin_factory.h"

namespace kwiver {
namespace vital {

const std::string plugin_factory::INTERFACE_TYPE( "interface-type" );
const std::string plugin_factory::CONCRETE_TYPE( "concrete-type" );
const std::string plugin_factory::PLUGIN_FILE_NAME( "plugin-file-name" );
const std::string plugin_factory::PLUGIN_NAME( "plugin-name" );
const std::string plugin_factory::PLUGIN_DESCRIPTION( "plugin-descrip" );


// ------------------------------------------------------------------
plugin_factory::
plugin_factory( std::string const& itype )
{
  m_interface_type = itype; // Optimize and store locally
  add_attribute( INTERFACE_TYPE, itype );
}


plugin_factory::
~plugin_factory()
{ }


// ------------------------------------------------------------------
bool plugin_factory::
get_attribute( std::string const& attr, std::string& val ) const
{
  auto const it = m_attribute_map.find( attr );
  if ( it != m_attribute_map.end() )
  {
    val = it->second;
    return true;
  }

  return false;
}


// ------------------------------------------------------------------
plugin_factory&
plugin_factory::
add_attribute( std::string const& attr, std::string const& val )
{
  // Create if not there. Overwrite if already there.
  m_attribute_map[attr] = val;

  return *this;
}

} } // end namespace
