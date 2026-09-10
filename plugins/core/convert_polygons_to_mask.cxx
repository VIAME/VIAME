/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "convert_polygons_to_mask.h"

#include <viame/algorithm_framework/util/transform_image.h>

#include <image_ops/polygon.h>

#include <algorithm>

namespace viame
{

void convert_polys_to_mask(
  const std::vector< std::string >& polygons,
  const kwiver::vital::bounding_box_d& bbox,
  kwiver::vital::image_of< uint8_t >& output )
{
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

    // Extract the x, y points from the split text, skipping '(poly)', and
    // shift them into the coordinates of the box
    viame::image_ops::polygon points;

    for( unsigned j = 1; j + 1 < poly_elements.size(); j += 2 )
    {
      points.emplace_back(
        std::stoi( poly_elements[ j ] ) - bbox_min_x,
        std::stoi( poly_elements[ j + 1 ] ) - bbox_min_y );
    }

    viame::image_ops::rasterize_polygon(
      points, output.width(), output.height(),
      [ &output ]( int x, int y )
      {
        output( x, y ) = 1;
      } );
  }

}

}
