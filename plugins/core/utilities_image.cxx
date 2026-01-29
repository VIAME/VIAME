/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "utilities_image.h"

#include <vital/types/image.h>

namespace viame {

// -------------------------------------------------------------------------------------------------
bool
is_non_8bit_image(
  const kwiver::vital::image_container_sptr& image,
  std::string& bit_depth_desc )
{
  if( !image )
  {
    return false;
  }

  auto pixel_traits = image->get_image().pixel_traits();

  if( pixel_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
  {
    if( pixel_traits.num_bytes == 2 )
    {
      bit_depth_desc = "16-bit unsigned";
      return true;
    }
    else if( pixel_traits.num_bytes > 2 )
    {
      bit_depth_desc = std::to_string( pixel_traits.num_bytes * 8 ) + "-bit unsigned";
      return true;
    }
    // 8-bit unsigned is the standard case
    return false;
  }
  else if( pixel_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
  {
    bit_depth_desc = std::to_string( pixel_traits.num_bytes * 8 ) + "-bit signed";
    return true;
  }
  else if( pixel_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
  {
    bit_depth_desc = std::to_string( pixel_traits.num_bytes * 8 ) + "-bit float";
    return true;
  }

  return false;
}


// -------------------------------------------------------------------------------------------------
std::string
get_image_bit_depth_description(
  const kwiver::vital::image_container_sptr& image )
{
  if( !image )
  {
    return "unknown";
  }

  auto pixel_traits = image->get_image().pixel_traits();

  std::string bits = std::to_string( pixel_traits.num_bytes * 8 );

  if( pixel_traits.type == kwiver::vital::image_pixel_traits::UNSIGNED )
  {
    return bits + "-bit unsigned";
  }
  else if( pixel_traits.type == kwiver::vital::image_pixel_traits::SIGNED )
  {
    return bits + "-bit signed";
  }
  else if( pixel_traits.type == kwiver::vital::image_pixel_traits::FLOAT )
  {
    return bits + "-bit float";
  }

  return "unknown";
}


} // end namespace viame
