// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core image_container interface
 */

#ifndef VITAL_IMAGE_CONTAINER_H_
#define VITAL_IMAGE_CONTAINER_H_

#include <vital/vital_config.h>

#include <vital/types/image.h>
#include <vital/types/metadata.h>

#include <vector>

namespace kwiver {
namespace vital {

/// An abstract representation of an image container.
/**
 * This class provides an interface for passing image data
 * between algorithms.  It is intended to be a wrapper for image
 * classes in third-party libraries and facilitate conversion between
 * various representations.  It provides limited access to the underlying
 * data and is not intended for direct use in image processing algorithms.
 */
class image_container
{
public:

  /// Destructor
  virtual ~image_container() = default;

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const = 0;

  /// The width of the image in pixels
  virtual size_t width() const = 0;

  /// The height of the image in pixels
  virtual size_t height() const = 0;

  /// The depth (or number of channels) of the image
  virtual size_t depth() const = 0;

  /// Get an in-memory image class to access the data
  virtual image get_image() const = 0;

  /// Get an in-memory image class to access a sub-image of the data
  virtual image get_image(unsigned x_offset, unsigned y_offset,
                          unsigned width, unsigned height) const
  {
    return get_image().crop(x_offset, y_offset, width, height);
  };

  /// Get metadata associated with this image
  virtual metadata_sptr get_metadata() const { return md_; }

  /// Set metadata associated with this image
  virtual void set_metadata(metadata_sptr md) { md_ = md; }

protected:
  /// optional metadata
  metadata_sptr md_;
};

/// Shared pointer for base image_container type
using image_container_sptr = std::shared_ptr< image_container >;
using image_container_scptr = std::shared_ptr< image_container const >;

/// List of image_container shared pointers
// NOTE(paul.tunison): This should be deprecated in favor of
//                     vital::image_container_set_sptr.
typedef std::vector<image_container_sptr> image_container_sptr_list;

// ==================================================================
/// This concrete image container is simply a wrapper around an image
class simple_image_container
: public image_container
{
public:

  /// Constructor
  explicit simple_image_container(const image& d, metadata_sptr m = nullptr)
  : data(d)
  {
    this->set_metadata(m);
  }

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const { return data.size(); }

  /// The width of the image in pixels
  virtual size_t width() const { return data.width(); }

  /// The height of the image in pixels
  virtual size_t height() const { return data.height(); }

  /// The depth (or number of channels) of the image
  virtual size_t depth() const { return data.depth(); }

  /// Get an in-memory image class to access the data
  virtual image get_image() const { return data; };

  /// Get an in-memory image class to access the data cropped
  virtual image get_image(unsigned x_offset, unsigned y_offset,
                          unsigned width, unsigned height) const
  {
    return data.crop(x_offset, y_offset, width, height);
  };

protected:
  /// data for this image container
  image data;
};

} } // end namespace vital

#endif // VITAL_IMAGE_CONTAINER_H_
