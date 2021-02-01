// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_CONVERT_IMAGE_H_
#define KWIVER_ARROWS_VISCL_CONVERT_IMAGE_H_

#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/algo/convert_image.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// Class to convert an image to a viscl base image
class KWIVER_ALGO_VISCL_EXPORT convert_image
  : public vital::algo::convert_image
{
public:

  /// Default Constructor
  convert_image();

  /// Image convert to viscl underlying type
  /**
   * \param [in] img image to be converted
   * \returns the image container with underlying viscl img
   * should be used to prevent repeated image uploading to GPU
   */
  virtual vital::image_container_sptr convert(vital::image_container_sptr img) const;
};

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
