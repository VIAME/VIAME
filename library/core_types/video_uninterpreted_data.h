// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Declaration of base video uninterpreted data type.

#ifndef VITAL_VIDEO_UNINTERPRETED_DATA_H_
#define VITAL_VIDEO_UNINTERPRETED_DATA_H_

#include <viame/core_types/vital_types_export.h>

#include <memory>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Base class for holding a single frame of uninterpreted data.
struct VITAL_TYPES_EXPORT video_uninterpreted_data
{
  virtual ~video_uninterpreted_data();
};

using video_uninterpreted_data_sptr =
  std::shared_ptr< video_uninterpreted_data >;
using video_uninterpreted_data_uptr =
  std::unique_ptr< video_uninterpreted_data >;

} // namespace vital

} // namespace kwiver

#endif
