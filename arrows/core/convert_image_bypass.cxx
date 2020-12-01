// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of Pass-through convert_image
 */

#include "convert_image_bypass.h"

namespace kwiver {
namespace arrows {
namespace core {

/// Default Constructor
convert_image_bypass
::convert_image_bypass()
{

}

/// Default image converter ( does nothing )
vital::image_container_sptr
convert_image_bypass
::convert(vital::image_container_sptr img) const
{
  return img;
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
