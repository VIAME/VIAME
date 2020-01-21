/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief VXL image container interface
 */

#ifndef KWIVER_ARROWS_VXL_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_VXL_IMAGE_CONTAINER_H_


#include <vital/vital_config.h>
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
