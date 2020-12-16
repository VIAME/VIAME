// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of VXL split image algorithm
 */

#include "split_image.h"

#include <arrows/vxl/image_container.h>

#include <vil/vil_crop.h>
#include <vil/vil_copy.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// Constructor
split_image
::split_image()
{
}

/// Destructor
split_image
::~split_image()
{
}

/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image
::split(kwiver::vital::image_container_sptr image) const
{
  std::vector< kwiver::vital::image_container_sptr > output;
  vil_image_view< vxl_byte > vxl_image = vxl::image_container::vital_to_vxl( image->get_image() );

  vil_image_view< vxl_byte > left_image_copy, left_image
    = vil_crop( vxl_image, 0, vxl_image.ni()/2, 0, vxl_image.nj() );
  vil_image_view< vxl_byte > right_image_copy, right_image
    = vil_crop( vxl_image, vxl_image.ni()/2, vxl_image.ni()/2, 0, vxl_image.nj() );

  vil_copy_deep( left_image, left_image_copy );
  vil_copy_deep( right_image, right_image_copy );

  output.push_back( vital::image_container_sptr( new vxl::image_container( left_image_copy ) ) );
  output.push_back( vital::image_container_sptr( new vxl::image_container( right_image_copy ) ) );

  return output;
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
