// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of OCV split image algorithm
 */

#include "split_image.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Constructor
split_image
::split_image()
{
}

/// Destructor
split_image
::~split_image()
{
}

/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image
::split(kwiver::vital::image_container_sptr image) const
{
  std::vector< kwiver::vital::image_container_sptr > output;
  cv::Mat cv_image = ocv::image_container::vital_to_ocv( image->get_image(), ocv::image_container::RGB_COLOR );
  cv::Mat left_image = cv_image( cv::Rect( 0, 0, cv_image.cols/2, cv_image.rows ) );
  cv::Mat right_image = cv_image( cv::Rect( cv_image.cols/2, 0, cv_image.cols/2, cv_image.rows ) );
  output.push_back( image_container_sptr( new ocv::image_container( left_image.clone(), ocv::image_container::RGB_COLOR ) ) );
  output.push_back( image_container_sptr( new ocv::image_container( right_image.clone(), ocv::image_container::RGB_COLOR ) ) );
  return output;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
