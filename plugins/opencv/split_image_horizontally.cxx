/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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
 * \brief Implementation of OCV split image horizontally algorithm
 */

#include "split_image_horizontally.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

/// Constructor
split_image_horizontally
::split_image_horizontally()
{
}

/// Destructor
split_image_horizontally
::~split_image_horizontally()
{
}

/// Split image
std::vector< kv::image_container_sptr >
split_image_horizontally
::split( kv::image_container_sptr image ) const
{
  std::vector< kv::image_container_sptr > output;

  cv::Mat cv_image =
    ocv::image_container::vital_to_ocv(
      image->get_image(), ocv::image_container::RGB_COLOR );

  cv::Mat left_image =
    cv_image(
      cv::Rect( 0, 0, cv_image.cols/2, cv_image.rows ) );
  cv::Mat right_image =
    cv_image(
      cv::Rect( cv_image.cols/2, 0, cv_image.cols/2, cv_image.rows ) );

  output.push_back(
    kv::image_container_sptr(
      new ocv::image_container( left_image.clone(), ocv::image_container::RGB_COLOR ) ) );
  output.push_back(
    kv::image_container_sptr(
      new ocv::image_container( right_image.clone(), ocv::image_container::RGB_COLOR ) ) );

  return output;
}

} // end namespace viame
