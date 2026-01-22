/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of OCV split image horizontally algorithm
 */

#include "split_image_horizontally.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

/// Constructor
split_image_horizontally
::split_image_horizontally()
{
}

/// Destructor
split_image_horizontally
::~split_image_horizontally()
{
}

/// Split image
std::vector< kv::image_container_sptr >
split_image_horizontally
::split( kv::image_container_sptr image ) const
{
  std::vector< kv::image_container_sptr > output;

  cv::Mat cv_image =
    ocv::image_container::vital_to_ocv(
      image->get_image(), ocv::image_container::RGB_COLOR );

  cv::Mat left_image =
    cv_image(
      cv::Rect( 0, 0, cv_image.cols/2, cv_image.rows ) );
  cv::Mat right_image =
    cv_image(
      cv::Rect( cv_image.cols/2, 0, cv_image.cols/2, cv_image.rows ) );

  output.push_back(
    kv::image_container_sptr(
      new ocv::image_container( left_image.clone(), ocv::image_container::RGB_COLOR ) ) );
  output.push_back(
    kv::image_container_sptr(
      new ocv::image_container( right_image.clone(), ocv::image_container::RGB_COLOR ) ) );

  return output;
}

} // end namespace viame
