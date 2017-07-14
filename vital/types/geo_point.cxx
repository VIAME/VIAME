/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief This file contains the implementation of a geo point.
 */

#include "geo_point.h"
#include "geodesy.h"

#include <iomanip>
#include <stdexcept>

namespace kwiver {
namespace vital {

using geo_raw_point_t = geo_point::geo_raw_point_t;

// ----------------------------------------------------------------------------
geo_point::
geo_point()
  : m_original_crs{ -1 }
{ }

// ----------------------------------------------------------------------------
geo_point::
geo_point( geo_raw_point_t const& point, int crs )
  : m_original_crs( crs )
{
  m_loc.insert( std::make_pair( crs, point ) );
}

// ----------------------------------------------------------------------------
bool geo_point
::is_empty() const
{
  return m_loc.empty();
}

// ----------------------------------------------------------------------------
geo_raw_point_t geo_point
::location() const
{
  return m_loc.at( m_original_crs );
}

// ----------------------------------------------------------------------------
int geo_point
::crs() const
{
  return m_original_crs;
}

// ----------------------------------------------------------------------------
geo_raw_point_t geo_point
::location( int crs ) const
{
  auto const i = m_loc.find( crs );
  if ( i == m_loc.end() )
  {
    auto const p = geo_conv( location(), m_original_crs, crs );
    m_loc.emplace( crs, p );
    return p;
  }

  return i->second;
}

// ----------------------------------------------------------------------------
void geo_point
::set_location( geo_raw_point_t const& loc, int crs )
{
  m_original_crs = crs;
  m_loc.clear();
  m_loc.insert( std::make_pair( crs, loc ) );
}

// ----------------------------------------------------------------------------
std::ostream&
operator<<( std::ostream& str, vital::geo_point const& obj )
{
  if ( obj.is_empty() )
  {
    str << "[ empty ]";
  }
  else
  {
    auto const old_prec = str.precision();
    auto const loc = obj.location();

    str << std::setprecision(22)
        << "[ " << loc[0]
        << " / " << loc[1]
        << " ] @ " << obj.crs();

    str.precision( old_prec );
  }

  return str;
}

} } // end namespace
