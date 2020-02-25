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
 * \brief OCV image_container inteface
 */

#ifndef KWIVER_ARROWS_OCV_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_OCV_IMAGE_CONTAINER_H_


#include <vital/vital_config.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <opencv2/core/core.hpp>

#include <vital/types/image_container.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// This image container wraps a cv::Mat
class KWIVER_ALGO_OCV_EXPORT image_container
  : public vital::image_container
{
public:

  enum ColorMode{RGB_COLOR, BGR_COLOR, OTHER_COLOR};

  /// Constructor - from a cv::Mat
  explicit image_container(const cv::Mat& d, ColorMode cm);

  /// Constructor - convert kwiver image to cv::Mat
  explicit image_container(const vital::image& vital_image)
  : data_(vital_to_ocv(vital_image, RGB_COLOR)) {}

  /// Constructor - convert base image container to cv::Mat
  explicit image_container(const vital::image_container& image_cont);

  /// Copy Constructor
  image_container(const arrows::ocv::image_container& other)
  : data_(other.data_) {}

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const;

  /// The width of the image in pixels
  virtual size_t width() const { return data_.cols; }

  /// The height of the image in pixels
  virtual size_t height() const { return data_.rows; }

  /// The depth (or number of channels) of the image
  virtual size_t depth() const { return data_.channels(); }

  /// Get an in-memory image class to access the data
  virtual vital::image get_image() const { return ocv_to_vital(data_, RGB_COLOR); }
  using vital::image_container::get_image;

  /// Access the underlying cv::Mat data structure
  cv::Mat get_Mat() const { return data_; }

  /// Convert an OpenCV cv::Mat to a VITAL image
  /**
   * This function constructs a vital::image from a cv::Mat and wraps the same
   * memory.  If the memory is owned by the cv::Mat this function will use a
   * mat_image_memory class to retain the original memory and reference
   * counting.  If the cv::Mat does not own the memory, the vital::image will
   * point to the same memory but also not take ownership.  That is
   * vital::image::memory() will return nullptr.
   */
  static vital::image ocv_to_vital(const cv::Mat& img, ColorMode cm);

  /// Convert an OpenCV cv::Mat type value to a vital::image_pixel_traits
  static vital::image_pixel_traits ocv_to_vital(int type);

  /// Convert a VITAL image to an OpenCV cv::Mat
  /**
  * This function constructs a cv::Mat from a vital::image and wraps the same
  * memory whenever possible.  If the vital::image contains a mat_image_memory
  * then the image data was originally created as a cv::Mat and this
  * function will reconstruct the cv::Mat using the original memory and
  * reference counting, if possible.  If the vital::image does not own its
  * memory, then the wrapped cv::Mat will also not own the memory.
  *
  * The supported memory layouts are far more restricted in cv::Mat than in
  * vital::image.  If the image is not of a compatible type and cannot be
  * directly wrapped then the a new cv::Mat is allocated and the image is
  * deep copied.  The same is true if the user requests a color mode other
  * than the default BGR used by OpenCV.  The data will be copied into new
  * memory and the color is converted.
  */
  static cv::Mat vital_to_ocv(const vital::image& img, ColorMode cm);

  /// Convert a vital::image_pixel_traits to an OpenCV cv::Mat type integer
  static int vital_to_ocv(const vital::image_pixel_traits& pt);

protected:
  /// image data
  cv::Mat data_;
};


/// Extract a cv::Mat from any image container
/**
 * If \a img is actually an arrows::ocv::image_container then
 * return the underlying cv::Mat.  Otherwise, convert the image data
 * to cv:Mat by shallow copy (if possible) or deep copy as a last resort.
 *
 * \param img Image container to convert to cv::mat
 */
KWIVER_ALGO_OCV_EXPORT cv::Mat image_container_to_ocv_matrix(const vital::image_container& img,
                           image_container::ColorMode cm = image_container::BGR_COLOR);


} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
