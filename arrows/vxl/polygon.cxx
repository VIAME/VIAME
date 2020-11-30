// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
