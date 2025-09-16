/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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

#include "convert_polygons_to_mask.h"

#include <vital/util/transform_image.h>

#ifdef VIAME_ENABLE_VXL
#include <vgl/vgl_polygon.h>
#include <vgl/vgl_polygon_scan_iterator.h>
#include <vgl/vgl_point_2d.h>
#endif

namespace viame
{

void convert_polys_to_mask(
  const std::vector< std::string >& polygons,
  const kwiver::vital::bounding_box_d& bbox,
  kwiver::vital::image_of< uint8_t >& output )
{
#ifdef VIAME_ENABLE_VXL
  if( polygons.empty() )
  {
    return;
  }

  // Get the box coordinates for later use
  int bbox_min_x = static_cast< int >( bbox.min_x() );
  int bbox_max_x = static_cast< int >( bbox.max_x() );
  int bbox_min_y = static_cast< int >( bbox.min_y() );
  int bbox_max_y = static_cast< int >( bbox.max_y() );

  size_t bbox_width = bbox_max_x - bbox_min_x;
  size_t bbox_height = bbox_max_y - bbox_min_y;

  // Create the mask as the size of the detection
  output = kwiver::vital::image_of< uint8_t >( bbox_width, bbox_height, 1 );

  // Set all the the data to 0
  transform_image( output, []( uint8_t ){ return 0; } );

  for( unsigned i = 0; i < polygons.size(); i++ )
  {
    // Split the last field by spaces
    std::vector< std::string > poly_elements;
    kwiver::vital::tokenize( polygons[i], poly_elements, " ", true );

    // Extract the x, y points from the split text, skipping '(poly)'
    std::vector< vgl_point_2d< double > > pts;
    for( unsigned j = 1; j < poly_elements.size(); j+=2 )
    {
      // Shift these points so they are in the coordinates of the box
      pts.push_back( vgl_point_2d< double >( std::stoi(poly_elements[j] ) -
        bbox_min_x, std::stoi( poly_elements[j+1]) - bbox_min_y ) );
    }
    // Create the polygon of the boundary
    vgl_polygon< double > poly = vgl_polygon< double >(
      pts.data(), static_cast< int >( pts.size() ) );

    // Create a scan iterator
    // x_min, x_max, y_min, y_max
    // Don't provide points outside this box
    vgl_box_2d< double > window( 0, bbox_width, 0, bbox_height );
    vgl_polygon_scan_iterator< double > psi( poly );

    for( psi.reset(); psi.next(); )
    {
      int y = psi.scany();

      // Make sure this is within the image
      if( y < 0 || y >= static_cast< int >( output.height() ) )
      {
        continue;
      }

      int min_x = std::max( 0, psi.startx() );
      int max_x = std::min( static_cast< int >( output.width() ) - 1, psi.endx() );

      for( int x = min_x; x <= max_x; ++x )
      {
        output( x, y ) = 1;
      }
    }
  }
#endif
}

}
