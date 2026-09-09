// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Declaration of base video raw metadata type.

#ifndef VITAL_VIDEO_RAW_METADATA_H_
#define VITAL_VIDEO_RAW_METADATA_H_

#include <viame/core_types/vital_types_export.h>

#include <memory>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Base class for holding a single frame of unprocessed metadata.
struct VITAL_TYPES_EXPORT video_raw_metadata
{
  virtual ~video_raw_metadata();
};

using video_raw_metadata_sptr = std::shared_ptr< video_raw_metadata >;
using video_raw_metadata_uptr = std::unique_ptr< video_raw_metadata >;

} // namespace vital

} // namespace kwiver

#endif
