// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
