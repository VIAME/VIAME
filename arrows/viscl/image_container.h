// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VISCL_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_VISCL_IMAGE_CONTAINER_H_

#include <vital/vital_config.h>
#include <arrows/viscl/kwiver_algo_viscl_export.h>

#include <vital/types/image_container.h>

#include <viscl/core/image.h>

namespace kwiver {
namespace arrows {
namespace vcl {

/// This image container wraps a VisCL image
class KWIVER_ALGO_VISCL_EXPORT image_container
: public vital::image_container
{
public:

  /// Constructor - from a VisCL image
  explicit image_container(const viscl::image& d)
  : data_(d) {}

  /// Constructor - convert vital image to VisCL image
  explicit image_container(const vital::image& vital_image)
  : data_(vital_to_viscl(vital_image)) {}

  /// Constructor - convert base image container to VisCL
  explicit image_container(const vital::image_container& image_cont);

  /// Copy Constructor
  image_container(const vcl::image_container& other)
  : data_(other.data_) {}

  /// The size of the image data in bytes
  /**
    * This size includes all allocated image memory,
    * which could be larger than width*height*depth.
    */
  virtual size_t size() const;

  /// The width of the image in pixels
  virtual size_t width() const { return data_.width(); }

  /// The height of the image in pixels
  virtual size_t height() const { return data_.height(); }

  /// The depth (or number of channels) of the image
  /**
    * viscl images only support 1 plane images at the moment
    */
  virtual size_t depth() const { return data_.depth(); }

  /// Get an in-memory image class to access the data
  virtual vital::image get_image() const { return viscl_to_vital(data_); }
  using vital::image_container::get_image;

  /// Access the underlying VisCL data structure
  viscl::image get_viscl_image() const { return data_; }

  /// Convert a VisCL image to a VITAL image
  static vital::image viscl_to_vital(const viscl::image& img);

  /// Convert a VITAL image to a VisCL image
  static viscl::image vital_to_viscl(const vital::image& img);

protected:

  viscl::image data_;
};

/// Extract a VisCL image from any image container
/**
 * If \a img is actually a vcl::image_container then
 * return the underlying VisCL image.  Otherwise, convert the image data
 * and upload to the GPU.
 */
KWIVER_ALGO_VISCL_EXPORT viscl::image image_container_to_viscl(const vital::image_container& img);

} // end namespace vcl
} // end namespace arrows
} // end namespace kwiver

#endif
