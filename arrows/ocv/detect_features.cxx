// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV detect_features algorithm implementation
 */

#include "detect_features.h"

#include <vector>

#include <opencv2/imgproc/imgproc.hpp>

#include <vital/exceptions/image.h>
#include <arrows/ocv/feature_set.h>
#include <arrows/ocv/image_container.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Extract a set of image features from the provided image
vital::feature_set_sptr
detect_features
::detect(vital::image_container_sptr image_data, vital::image_container_sptr mask) const
{
  cv::Mat cv_img = image_container::vital_to_ocv(image_data->get_image(), ocv::image_container::BGR_COLOR );
  cv::Mat cv_mask;
  std::vector<cv::KeyPoint> keypoints;

  // Only initialize a mask image if the given mask image container contained
  // valid data.
  if( mask && mask->size() > 0 )
  {
    if ( image_data->width() != mask->width() ||
         image_data->height() != mask->height() )
    {
      VITAL_THROW( image_size_mismatch_exception,
          "OCV detect feature algorithm given a non-zero mask with mismatched "
          "shape compared to input image",
          image_data->width(), image_data->height(),
          mask->width(), mask->height()
          );
    }

    // Make sure we make a one-channel cv::Mat
    vital::image s = mask->get_image();
    // hijacking memory of given mask image, but only telling the new image
    // object to consider the first channel. See vital::image documentation.
    vital::image i(s.memory(),
                   s.first_pixel(),
                   s.width(),  s.height(), 1 /*depth*/,
                   s.w_step(), s.h_step(), s.d_step(), s.pixel_traits());
    cv_mask = ocv::image_container::vital_to_ocv(i, ocv::image_container::BGR_COLOR);
    cv::threshold(cv_mask, cv_mask, 128, 255, cv::THRESH_BINARY);
  }

  detector->detect(cv_img, keypoints, cv_mask);
  return feature_set_sptr(new feature_set(keypoints));
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
