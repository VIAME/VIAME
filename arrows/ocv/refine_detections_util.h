// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Common utilities for refine_detections implementations

#ifndef KWIVER_ARROWS_OCV_REFINE_DETECTIONS_UTIL_H_
#define KWIVER_ARROWS_OCV_REFINE_DETECTIONS_UTIL_H_

#include <opencv2/core/core.hpp>

#include <vital/types/bounding_box.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// Convert a bounding_box_d to the Rect that should describe its mask
cv::Rect bbox_to_mask_rect( kwiver::vital::bounding_box_d const& bbox );

}
}
}
#endif
