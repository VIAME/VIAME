/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief GDAL image_container inteface
 */

#ifndef KWIVER_ARROWS_GDAL_IMAGE_CONTAINER_H_
#define KWIVER_ARROWS_GDAL_IMAGE_CONTAINER_H_


#include <vital/vital_config.h>
#include <arrows/gdal/kwiver_algo_gdal_export.h>

#include <vital/types/image_container.h>

namespace kwiver {
namespace arrows {
namespace gdal {

/// This image container wraps a cv::Mat
class KWIVER_ALGO_GDAL_EXPORT image_container
  : public vital::image_container
{
public:

  /// Constructor
  explicit image_container();

  /// The size of the image data in bytes
  /**
   * This size includes all allocated image memory,
   * which could be larger than width*height*depth.
   */
  virtual size_t size() const { return 1000; }

  /// The width of the image in pixels
  virtual size_t width() const { return 1000; }

  /// The height of the image in pixels
  virtual size_t height() const { return 1000; }

  /// The depth (or number of channels) of the image
  virtual size_t depth() const { return 3; }

  /// Get and in-memory image class to access the data
  virtual vital::image get_image() const { return data; }

protected:

  vital::image data;
};



} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_GDAL_IMAGE_CONTAINER_H_
