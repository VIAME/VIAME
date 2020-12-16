// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the convert_image algorithm that acts as a bypass
 */

#ifndef KWIVER_ARROWS_CORE_CONVERT_IMAGE_BYPASS_H_
#define KWIVER_ARROWS_CORE_CONVERT_IMAGE_BYPASS_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/convert_image.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for bypassing image conversion
class KWIVER_ALGO_CORE_EXPORT convert_image_bypass
  : public vital::algo::convert_image
{
public:
  PLUGIN_INFO( "bypass",
               "Performs no conversion and returns the given image container." )

   /// Default Constructor
  convert_image_bypass();

  /// Default image converter ( does nothing )
  /**
   * \param [in] img image to be converted
   * \returns the input image
   */
  virtual vital::image_container_sptr convert(vital::image_container_sptr img) const;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
