// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of habcam split image horizontally algorithm
 */

#include "split_image_habcam.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace kv = kwiver::vital;

namespace viame {

/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image_habcam
::split( kwiver::vital::image_container_sptr image ) const
{
  std::vector< kwiver::vital::image_container_sptr > output;

  if( image->width() >= c_required_width_factor * image->height() )
  {
    cv::Mat cv_image =
      kwiver::arrows::ocv::image_container::vital_to_ocv(
        image->get_image(),
        kwiver::arrows::ocv::image_container::RGB_COLOR );

    cv::Mat left_image =
      cv_image(
        cv::Rect( 0, 0, cv_image.cols / 2, cv_image.rows ) );
    cv::Mat right_image =
      cv_image(
        cv::Rect( cv_image.cols / 2, 0, cv_image.cols / 2, cv_image.rows ) );

    output.push_back(
      kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container( left_image.clone(),
        kwiver::arrows::ocv::image_container::RGB_COLOR ) ) );
    output.push_back(
      kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container( right_image.clone(),
        kwiver::arrows::ocv::image_container::RGB_COLOR ) ) );
  }
  else
  {
    output.push_back( image );
  }

  return output;
}

} // end namespace viame
