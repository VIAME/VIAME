// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Common utilities for refine_detections implementations

#ifndef VIAME_OPENCV_REFINE_DETECTIONS_UTIL_H
#define VIAME_OPENCV_REFINE_DETECTIONS_UTIL_H

#include <opencv2/core/core.hpp>

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object.h>

namespace viame {

/// Convert a bounding_box_d to the Rect that should describe its mask
cv::Rect bbox_to_mask_rect( kwiver::vital::bounding_box_d const& bbox );

/// Get the mask associated with det, adjusted to the expected size
///
/// The expected size is \code bbox_to_mask_rect( det->bounding_box() ).size() \endcode
cv::Mat get_standard_mask( kwiver::vital::detected_object_sptr const& det );

}

#endif
