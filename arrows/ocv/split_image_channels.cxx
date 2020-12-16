// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of OCV split image algorithm
 */

#include "split_image_channels.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Constructor
split_image_channels
::split_image_channels()
{
}

/// Destructor
split_image_channels
::~split_image_channels()
{
}

/// Split image
std::vector< kwiver::vital::image_container_sptr >
split_image_channels
::split( kwiver::vital::image_container_sptr image ) const
{
  std::vector< kwiver::vital::image_container_sptr > output;

  cv::Mat cv_image = ocv::image_container::vital_to_ocv(
    image->get_image(), ocv::image_container::RGB_COLOR );

  std::vector< cv::Mat > channels;
  cv::split( cv_image, channels );

  for( auto channel : channels )
  {
    output.push_back(
      image_container_sptr(
        new ocv::image_container(
          channel.clone(), ocv::image_container::RGB_COLOR ) ) );
  }

  return output;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
