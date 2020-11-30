// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of OCV merge images algorithm
 */

#include "merge_images.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Constructor
merge_images
::merge_images()
{
}

/// Merge images
kwiver::vital::image_container_sptr
merge_images::merge(kwiver::vital::image_container_sptr image1,
                    kwiver::vital::image_container_sptr image2) const
{
  cv::Mat cv_image1 = ocv::image_container::vital_to_ocv(image1->get_image(),
    ocv::image_container::RGB_COLOR);
  cv::Mat cv_image2 = ocv::image_container::vital_to_ocv(image2->get_image(),
    ocv::image_container::RGB_COLOR);
  cv::Mat fin_image;

  std::vector<cv::Mat> channels1, channels2, all_channels;
  cv::split(cv_image1, channels1);
  cv::split(cv_image2, channels2);
  all_channels.insert(all_channels.begin(), channels1.begin(), channels1.end());
  all_channels.insert(all_channels.end(), channels2.begin(), channels2.end());

  cv::merge(all_channels, fin_image);

  kwiver::vital::image_container_sptr concatenated_image_container =
      image_container_sptr(new ocv::image_container(fin_image.clone(),
                           ocv::image_container::RGB_COLOR));

  return concatenated_image_container;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
