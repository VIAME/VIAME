// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation for the geo MGRS coordinate
 */

#include "geo_MGRS.h"

#include <iomanip>

namespace kwiver {
namespace vital {

geo_MGRS
::geo_MGRS()
{ }

geo_MGRS
::geo_MGRS(std::string const& coord)
: mgrs_coord_(coord)
{ }

geo_MGRS
::~geo_MGRS()
{ }

bool
geo_MGRS
::is_empty() const
{
  return this->mgrs_coord_.empty();
}

bool
geo_MGRS
::is_valid() const
{
  if (is_empty())
  {
    return false;
  }

  // TODO - what constitutes a valid MGRS?
  return true;
}

geo_MGRS &
geo_MGRS
::set_coord( std::string const& coord)
{
  this-> mgrs_coord_ = coord;
  return *this;
}

std::string const&
geo_MGRS
::coord() const
{
  return this->mgrs_coord_;
}

bool
geo_MGRS
::operator == ( const geo_MGRS &rhs ) const
{
  // May want to take into precision of operands.
  return ( rhs.coord() == this->coord() );
}

bool
geo_MGRS
::operator != ( const geo_MGRS &rhs ) const
{
  return ( !( this->operator == ( rhs ) ) );
}

geo_MGRS
geo_MGRS
::operator=( const geo_MGRS& m )
{
  if ( this != & m )
  {
    this->mgrs_coord_ = m.coord();
  }

  return *this;
}

std::ostream & operator<< (std::ostream & str, const kwiver::vital::geo_MGRS & obj)
{
  str << "[MGRS: " << obj.coord() << "]";

  return str;
}

} // end namespace
} // end namespace
