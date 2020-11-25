// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL image container interface
 */

#ifndef KWIVER_ARROWS_VXL_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_VXL_IMAGE_CONTAINER_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/image_container.h>

#include <vil/vil_image_view.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// This image container wraps a vil_image_view
/**
 * This class represents an image using vil_image_view format to store
 * the image data by extending the basic image_container.
 */
class KWIVER_ALGO_VXL_EXPORT image_container
  : public vital::image_container
{
public:

  /// Constructor - from a vil_image_view_base
  explicit image_container(const vil_image_view_base& d);

  /// Constructor - from a vil_image_view_base_sptr
  explicit image_container(const vil_image_view_base_sptr d)
  : data_(d) {}

  /// Constructor - convert vital image to vil
  explicit image_container(const vital::image& vital_image)
  : data_(vital_to_vxl(vital_image)) {}

  /// Constructor - convert base image container to vil
  explicit image_container(const vital::image_container& image_cont);

  /// Copy Constructor
  image_container(const arrows::vxl::image_container& other)
  : data_(other.data_) {}

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const;

  /// The width of the image in pixels
  virtual size_t width() const { return data_->ni(); }

  /// The height of the image in pixels
  virtual size_t height() const { return data_->nj(); }

  /// The depth (or number of channels) of the image
  virtual size_t depth() const { return data_->nplanes(); }

  /// Get an in-memory image class to access the data
  virtual vital::image get_image() const { return vxl_to_vital(*data_); }
  using vital::image_container::get_image;

  /// Get image data in this container.
  vil_image_view_base_sptr get_vil_image_view() const { return data_; }

  /// Convert a VXL vil_image_view to a VITAL image
  static vital::image vxl_to_vital(const vil_image_view_base& img);

  /// Convert a VITAL image to a VXL vil_image_view
  static vil_image_view_base_sptr vital_to_vxl(const vital::image& img);

protected:
  /// image data
  vil_image_view_base_sptr data_;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
