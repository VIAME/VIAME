// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_WARP_IMAGE_H_
#define VITAL_ALGO_WARP_IMAGE_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/homography.h>
#include <viame/core_types/image.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
/// An abstract base class for warping an image onto another image.
class VITAL_ALGO_EXPORT warp_image
  : public algorithm
{
public:
  warp_image();

  PLUGGABLE_INTERFACE( warp_image );

  /// Warp \p src_image onto \p dst_image.
  ///
  /// \param src_image Source image to draw pixel values from.
  /// \param dst_image Destination image to draw pixel values to.
  /// \param homography
  ///   Homography mapping \p src_image to \p dst_image, in pixels.
  /// \param alpha_mask
  ///   Optional single-channel image indicating the opacity of \p src_image.
  ///
  /// \return
  ///   Result after warping. This may be \p dst_image or a new image object.
  ///   Implementations are encouraged to perform the operation in-place
  ///   (returning the modified \p dst_image ) if possible.
  virtual image_container_sptr warp(
    image_container_sptr src_image,
    image_container_sptr dst_image,
    homography_sptr homography,
    image_container_sptr alpha_mask = nullptr ) const = 0;
};

using warp_image_sptr = std::shared_ptr< warp_image >;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
