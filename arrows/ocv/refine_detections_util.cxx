// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Common utilities for refine_detections implementations

#include "refine_detections_util.h"

#include <cmath>

namespace kwiver {
namespace arrows {
namespace ocv {

cv::Rect bbox_to_mask_rect( kwiver::vital::bounding_box_d const& bbox )
{
  auto min_x = static_cast< int >( std::floor( bbox.min_x() ) );
  auto min_y = static_cast< int >( std::floor( bbox.min_y() ) );
  return cv::Rect( min_x, min_y,
                   static_cast< int >( std::ceil( bbox.max_x() ) ) - min_x,
                   static_cast< int >( std::ceil( bbox.max_y() ) ) - min_y );
}

}
}
}
