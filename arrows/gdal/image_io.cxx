// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief GDAL image_io implementation
 */

#include "image_io.h"

#include <arrows/gdal/image_container.h>

#include <vital/exceptions/algorithm.h>
#include <vital/vital_config.h>

namespace kwiver {
namespace arrows {
namespace gdal {

/// Load image image from the file
/**
 * \param filename the path to the file the load
 * \returns an image container refering to the loaded image
 */
vital::image_container_sptr
image_io
::load_(const std::string& filename) const
{
  return vital::image_container_sptr( new gdal::image_container( filename ) );
}

/// Save image image to a file
/**
 * \param filename the path to the file to save.
 * \param data The image container refering to the image to write.
 */
void
image_io
::save_( VITAL_UNUSED const std::string& filename,
         VITAL_UNUSED vital::image_container_sptr data) const
{
  VITAL_THROW( vital::algorithm_exception, this->type_name(), this->impl_name(),
               "Saving to file not supported." );
}

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver
