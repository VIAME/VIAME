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
 * \brief This file contains the implementation of a geo polygon.
 */

#include "geo_polygon.h"

#include <stdexcept>

namespace kwiver {
namespace vital {

using geo_raw_polygon_t = geo_polygon::geo_raw_polygon_t;

// ----------------------------------------------------------------------------
geo_polygon::
geo_polygon()
  : m_original_crs{ -1 }
{ }

// ----------------------------------------------------------------------------
geo_polygon::
geo_polygon( geo_raw_polygon_t const& polygon, int crs )
  : m_original_crs( crs )
{
  m_poly.insert( std::make_pair( crs, polygon ) );
}

// ----------------------------------------------------------------------------
bool geo_polygon
::is_empty() const
{
  return m_poly.empty();
}

// ----------------------------------------------------------------------------
geo_raw_polygon_t geo_polygon
::polygon() const
{
  return m_poly.at( m_original_crs );
}

// ----------------------------------------------------------------------------
int geo_polygon
::crs() const
{
  return m_original_crs;
}

// ----------------------------------------------------------------------------
geo_raw_polygon_t geo_polygon
::polygon( int crs ) const
{
  auto const i = m_poly.find( crs );
  if ( i == m_poly.end() )
  {
    // TODO convert to requested CRS and add to m_poly
    throw std::runtime_error( "conversion is not implemented" );
  }

  return i->second;
}

// ----------------------------------------------------------------------------
void geo_polygon
::set_polygon( geo_raw_polygon_t const& poly, int crs )
{
  m_original_crs = crs;
  m_poly.clear();
  m_poly.insert( std::make_pair( crs, poly ) );
}

} } // end namespace
