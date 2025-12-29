// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Common utilities for refine_detections implementations

#include "refine_detections_util.h"

#include <cmath>

#include <arrows/ocv/image_container.h>

namespace viame {

namespace ocv = kwiver::arrows::ocv;

cv::Rect bbox_to_mask_rect( kwiver::vital::bounding_box_d const& bbox )
{
  auto min_x = static_cast< int >( std::floor( bbox.min_x() ) );
  auto min_y = static_cast< int >( std::floor( bbox.min_y() ) );
  return cv::Rect( min_x, min_y,
                   static_cast< int >( std::ceil( bbox.max_x() ) ) - min_x,
                   static_cast< int >( std::ceil( bbox.max_y() ) ) - min_y );
}

cv::Mat get_standard_mask( kwiver::vital::detected_object_sptr const& det )
{
  auto vital_mask = det->mask();
  if( !vital_mask )
  {
    return {};
  }
  using ic = ocv::image_container;
  cv::Mat mask = ic::vital_to_ocv( vital_mask->get_image(), ic::OTHER_COLOR );
  auto size = bbox_to_mask_rect( det->bounding_box() ).size();
  if( mask.size() == size )
  {
    return mask;
  }
  cv::Mat standard_mask( size, CV_8UC1, cv::Scalar( 0 ) );
  cv::Rect intersection( 0, 0,
                         std::min( size.width, mask.cols ),
                         std::min( size.height, mask.rows ) );
  mask( intersection ).copyTo( standard_mask( intersection ) );
  return standard_mask;
}

} // end namespace viame
