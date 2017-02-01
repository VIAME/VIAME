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
 * \brief vxl polygon conversion functions
 */

#include "polygon.h"

#include <stdexcept>

namespace kwiver {
namespace arrows {
namespace vxl {

// ------------------------------------------------------------------
kwiver::vital::polygon_sptr
vxl_to_vital( const vgl_polygon< double >& poly )
{
  if ( poly.num_sheets() != 1 )
  {
    throw std::out_of_range( "vgl_polygon is not a single sheet polygon." );
  }

  // Convert vxl polygon to vital format
  auto local_poly = std::make_shared< kwiver::vital::polygon > ();

  const vgl_polygon< double >::sheet_t& sheet = poly[0];
  const size_t limit = sheet.size();
  for ( size_t i = 0; i < limit; ++i )
  {
    auto pt = sheet[i];
    local_poly->push_back( pt.x(), pt.y() );
  } // end for

  return local_poly;
}


// ------------------------------------------------------------------
std::shared_ptr< vgl_polygon< double > >
vital_to_vxl( kwiver::vital::polygon_sptr poly )
{
  // Convert vital type polygon to VXL format
  auto local_poly  = std::make_shared< vgl_polygon< double > > ( 1 );
  size_t limit = poly->num_vertices();

  for ( size_t i = 0; i < limit; ++i )
  {
    auto pt = poly->at( i );
    local_poly->push_back( pt[0], pt[1] );
  } // end for

  return local_poly;
}


}
}
}         // end namespace
