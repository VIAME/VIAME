// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OpenCV image_io implementation
 */

#include "image_io.h"

#include <arrows/ocv/image_container.h>

#include <vital/types/metadata_traits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/// Load image image from the file
/**
 * \param filename the path to the file to load
 * \returns an image container refering to the loaded image
 */
vital::image_container_sptr
image_io
::load_(const std::string& filename) const
{
  auto md = std::make_shared<kwiver::vital::metadata>();
  md->add<kwiver::vital::VITAL_META_IMAGE_URI>(filename);

  cv::Mat img = cv::imread(filename.c_str(), -1);
  auto img_ptr = vital::image_container_sptr(new ocv::image_container(img, ocv::image_container::BGR_COLOR));
  img_ptr->set_metadata(md);
  return img_ptr;
}

/// Save image image to a file
/**
 * \param filename the path to the file to save.
 * \param data The image container refering to the image to write.
 */
void
image_io
::save_(const std::string& filename,
       vital::image_container_sptr data) const
{
  cv::Mat img = ocv::image_container::vital_to_ocv(data->get_image(), ocv::image_container::BGR_COLOR);
  cv::imwrite(filename.c_str(), img);
}

/// Load image metadata from the file
/**
 * \param filename the path to the file to read
 * \returns pointer to the loaded metadata
 */
kwiver::vital::metadata_sptr
image_io
::load_metadata_(const std::string& filename) const
{
  auto md = std::make_shared<kwiver::vital::metadata>();
  md->add<kwiver::vital::VITAL_META_IMAGE_URI>(filename);
  return md;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
