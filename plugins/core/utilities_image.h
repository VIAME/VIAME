/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_IMAGE_H
#define VIAME_CORE_UTILITIES_IMAGE_H

#include "viame_core_export.h"

#include <vital/types/image_container.h>

#include <string>

namespace viame {

/// @brief Check if an image is non-8-bit (16-bit, signed, or floating point).
///
/// This function examines the pixel traits of an image to determine if it
/// uses a format other than standard 8-bit unsigned. This is useful for
/// detecting imagery that may need normalization before use with models
/// that expect 8-bit input.
///
/// @param image The image container to check
/// @param bit_depth_desc Output parameter filled with a human-readable
///        description of the bit depth (e.g., "16-bit unsigned", "32-bit float")
///        Only set if the function returns true.
/// @return true if the image is non-8-bit, false if it is standard 8-bit unsigned
VIAME_CORE_EXPORT
bool is_non_8bit_image(
  const kwiver::vital::image_container_sptr& image,
  std::string& bit_depth_desc );

/// @brief Get a human-readable description of image pixel format.
///
/// @param image The image container to describe
/// @return A string describing the pixel format (e.g., "8-bit unsigned",
///         "16-bit unsigned", "32-bit float")
VIAME_CORE_EXPORT
std::string get_image_bit_depth_description(
  const kwiver::vital::image_container_sptr& image );

} // end namespace viame

#endif /* VIAME_CORE_UTILITIES_IMAGE_H */
