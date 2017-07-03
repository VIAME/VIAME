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
