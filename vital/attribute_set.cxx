// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
