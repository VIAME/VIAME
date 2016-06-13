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

  // TODO - what constututes a valid MGRS?
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
