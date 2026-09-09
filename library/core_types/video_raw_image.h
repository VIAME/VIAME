// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Declaration of base video raw image type.

#ifndef VITAL_VIDEO_RAW_IMAGE_H_
#define VITAL_VIDEO_RAW_IMAGE_H_

#include <viame/core_types/vital_types_export.h>

#include <memory>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Base class for holding a single frame of unprocessed image data.
struct VITAL_TYPES_EXPORT video_raw_image
{
  virtual ~video_raw_image();
};

using video_raw_image_sptr = std::shared_ptr< video_raw_image >;
using video_raw_image_uptr = std::unique_ptr< video_raw_image >;

} // namespace vital

} // namespace kwiver

#endif
