// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of OCV DescriptorExtractor wrapping.
 */

#include "extract_descriptors.h"

#include <arrows/ocv/image_container.h>
#include <arrows/ocv/feature_set.h>
#include <arrows/ocv/descriptor_set.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Extract from the image a descriptor corresponding to each feature
descriptor_set_sptr
extract_descriptors
::extract(image_container_sptr image_data,
          feature_set_sptr &features,
          image_container_sptr /* image_mask */) const
{
  if( !image_data || !features )
  {
    return descriptor_set_sptr();
  }
  cv::Mat img = image_container_to_ocv_matrix(*image_data, ocv::image_container::BGR_COLOR);
  std::vector<cv::KeyPoint> kpts = features_to_ocv_keypoints(*features);

  cv::Mat desc;
  cv::Mat mask;
  extractor->detectAndCompute( img, mask, kpts, desc, true);

  //keypoint order may have changed.  We must output keypoints.
  features = std::make_shared<feature_set>(kpts);

  return descriptor_set_sptr(new ocv::descriptor_set(desc));
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
