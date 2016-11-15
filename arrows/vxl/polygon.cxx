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
 * \brief vxl polygon implementation
 */

#include "polygon.h"

#include <stdexcept>
#include <sstream>
#include <memory>

namespace kwiver {
namespace arrows {
namespace vxl {

// ------------------------------------------------------------------
polygon::
polygon()
  : m_polygon( 1 ) // we only support one sheet
{ }


// ------------------------------------------------------------------
polygon::
polygon( const vgl_polygon< double >& poly )
  : m_polygon( poly )
{
}


// ------------------------------------------------------------------
polygon::
~polygon()
{ }


// ------------------------------------------------------------------
void
polygon::
push_back( double x, double y )
{
  m_polygon.push_back( x, y );
}


// ------------------------------------------------------------------
void
polygon::
push_back( const point_t& pt )
{
  m_polygon.push_back( pt[0], pt[1] );
}


// ------------------------------------------------------------------
bool
polygon::
contains( double x, double y )
{
  return m_polygon.contains( x, y );
}


// ------------------------------------------------------------------
size_t
polygon::
num_vertices() const
{
  return m_polygon.num_vertices();
}


// ------------------------------------------------------------------
std::vector< kwiver::vital::polygon::point_t >
polygon::
get_vertices() const
{
  if ( m_polygon.num_sheets() < 1 )
  {
    throw std::out_of_range( "vgl_polygon does not any sheets. It is empty" );
  }

  const size_t limit = m_polygon.num_vertices();
  std::vector< kwiver::vital::polygon::point_t > retval( limit );

  const vgl_polygon<double>::sheet_t& sheet = m_polygon[0];
  for ( size_t i = 0; i < limit; ++i )
  {
    auto pt = sheet[i];
    retval.push_back( kwiver::vital::polygon::point_t( pt.x(), pt.y() ) );
  } // end for

  return retval;
}


// ------------------------------------------------------------------
bool
polygon::
contains( const point_t& pt )
{
  return m_polygon.contains( pt[0], pt[1] );
}


// ------------------------------------------------------------------
kwiver::vital::polygon::point_t
polygon::
at( size_t idx ) const
{
  if ( m_polygon.num_sheets() < 1 )
  {
    throw std::out_of_range( "vgl_polygon does not any sheets. It is empty" );
  }

  auto sheet_0 = m_polygon[0];

  if ( idx >= sheet_0.size() )
  {
    std::stringstream str;
    str << "Requested index " << idx
        << " is beyond the end of the polygon. Last valid index is "
        << sheet_0.size() - 1;
    throw std::out_of_range( str.str() );
  }

  kwiver::vital::polygon::point_t retval;
  retval[0] = sheet_0[idx].x();
  retval[1] = sheet_0[idx].y();
  return retval;
}


// ------------------------------------------------------------------
kwiver::vital::vital_polygon_sptr
polygon::
get_polygon()
{
  // Convert vxl polygon to vital format
  auto local_poly = new kwiver::vital::vital_polygon();

  const size_t limit = m_polygon.num_vertices();
  const vgl_polygon<double>::sheet_t& sheet = m_polygon[0];
  for ( size_t i = 0; i < limit; ++i )
  {
    auto pt = sheet[i];
    local_poly->push_back( pt.x(), pt.y() );
  } // end for
  return kwiver::vital::vital_polygon_sptr( local_poly );
}


// ------------------------------------------------------------------
kwiver::arrows::vxl::polygon_sptr
polygon::
get_vxl_polygon( kwiver::vital::polygon_sptr poly )
{
  if ( dynamic_cast< kwiver::arrows::vxl::polygon* > ( poly.get() ) )
  {
    // Return derived class pointer with same usage count.
    return std::dynamic_pointer_cast< kwiver::arrows::vxl::polygon >(poly);
  }

  // Convert vital type polygon to VXL format
  vgl_polygon< double > local_poly(1);
  size_t limit = poly->num_vertices();
  for ( size_t i = 0; i < limit; ++i )
  {
    auto pt = poly->at(i);
    local_poly.push_back( pt[0], pt[1] );
  } // end for

  // Create returned object
  return std::make_shared< kwiver::arrows::vxl::polygon > ( local_poly );
}

} } }     // end namespace
