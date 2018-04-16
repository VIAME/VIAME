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

/**
 * \file
 * \brief Implementation of attribute_set class
 */

#include "attribute_set.h"

#include <sstream>


namespace kwiver {
namespace vital {

// ==================================================================
attribute_set_exception::
attribute_set_exception( std::string const& str )
{
  m_what = str;
}


attribute_set_exception::
~attribute_set_exception() noexcept
{ }



// ==================================================================
attribute_set::
attribute_set()
{ }


attribute_set::
~attribute_set()
{ }


// ------------------------------------------------------------------
attribute_set_sptr
attribute_set::
clone() const
{
  auto new_obj = std::make_shared< attribute_set >();
  auto it( this->m_attr_map.begin() );
  const auto eit( this->m_attr_map.end() );
  for ( ; it != eit; ++it)
  {
    new_obj->add( it->first, *it->second );
  }

  return new_obj;
}


// ------------------------------------------------------------------
void
attribute_set::
add( const std::string& name, const kwiver::vital::any& val )
{
#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  m_attr_map[name] = std::make_unique<kwiver::vital::any>(val);
#else
  m_attr_map[name] = std::make_shared<kwiver::vital::any>(val);
#endif
}


// ------------------------------------------------------------------
bool attribute_set::
has( const std::string& name ) const
{
  return m_attr_map.count( name ) > 0;
}


// ------------------------------------------------------------------
bool
attribute_set::
erase( const std::string& name )
{
  return m_attr_map.erase(name) > 0;
}


// ------------------------------------------------------------------
attribute_set::const_iterator_t
attribute_set::
begin() const
{
  return m_attr_map.begin();
}


// ------------------------------------------------------------------
attribute_set::const_iterator_t
attribute_set::
end() const
{
  return m_attr_map.end();
}


// ------------------------------------------------------------------
size_t
attribute_set::
size() const
{
  return m_attr_map.size();
}


// ------------------------------------------------------------------
bool
attribute_set::
empty() const
{
  return m_attr_map.empty();
}


// ------------------------------------------------------------------
kwiver::vital::any
attribute_set::
data( const std::string& name ) const
{
  auto ix = m_attr_map.find( name );
  if ( ix == m_attr_map.end() )
  {
    std::stringstream str;
    str << "Attribute name \"" << name << "\" is not in the set.";
    VITAL_THROW( attribute_set_exception, str.str() );
  }

  return *(ix->second);
}

} } // end namespace
