// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "convert_image.h"

#include <arrows/viscl/image_container.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Default Constructor
convert_image
::convert_image()
{

}

/// Image convert to viscl underlying type
vital::image_container_sptr
convert_image
::convert(vital::image_container_sptr img) const
{
  // make new viscl image container
  return std::shared_ptr<image_container>(new image_container(*img));
}

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver
