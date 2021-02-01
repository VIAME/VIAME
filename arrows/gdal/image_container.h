// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief GDAL image_container inteface
 */

#ifndef KWIVER_ARROWS_GDAL_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_GDAL_IMAGE_CONTAINER_H_

#include <vital/vital_config.h>
#include <arrows/gdal/kwiver_algo_gdal_export.h>

#include <vital/types/image_container.h>

#include <gdal_priv.h>

namespace kwiver {
namespace arrows {
namespace gdal {

/// This image container wraps a cv::Mat
class KWIVER_ALGO_GDAL_EXPORT image_container
  : public vital::image_container
{
public:

  /// Constructor - from file
  explicit image_container(const std::string& filename);

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const;

  /// The width of the image in pixels
  virtual size_t width() const
  { return static_cast<size_t>(gdal_dataset_->GetRasterXSize()); }

  /// The height of the image in pixels
  virtual size_t height() const
  { return static_cast<size_t>(gdal_dataset_->GetRasterYSize()); }

  /// The depth (or number of channels) of the image
  virtual size_t depth() const
  { return static_cast<size_t>(gdal_dataset_->GetRasterCount()); }

  /// Get image. Unlike other image containers must allocate memory
  virtual vital::image get_image() const;

  /// Get cropped view of image. Unlike other image containers must allocate memory
  virtual vital::image get_image(unsigned x_offset, unsigned y_offset,
                                 unsigned width, unsigned height) const;

  char **get_raw_metadata_for_domain(const char *domain);
protected:

  std::shared_ptr<GDALDataset> gdal_dataset_;
  vital::image_pixel_traits pixel_traits_;
};

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver

#endif
