/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "windowed_detector_resize.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <exception>

namespace kwiver {
namespace arrows {
namespace ocv {


double
scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst,
                            int width, int height, bool pad )
{
  double scale = 1.0;

  if( src.rows == height && src.cols == width )
  {
    dst = src;
    return scale;
  }

  double original_height = static_cast< double >( src.rows );
  double original_width = static_cast< double >( src.cols );

  if( original_height > height )
  {
    scale = height / original_height;
  }
  if( original_width > width )
  {
    scale = std::min( scale, width / original_width );
  }

  cv::Mat resized;
  cv::resize( src, resized, cv::Size(), scale, scale );

  if( pad )
  {
    dst.create( height, width, src.type() );
    dst.setTo( 0 );

    cv::Rect roi( 0, 0, resized.cols, resized.rows );
    cv::Mat aoi( dst, roi );

    resized.copyTo( aoi );
  }
  else
  {
    dst = resized;
  }

  return scale;
}

double
format_image( const cv::Mat& src, cv::Mat& dst, std::string option,
              double scale_factor, int width, int height, bool pad )
{
  double scale = 1.0;

  if( option == "maintain_ar" )
  {
    scale = scale_image_maintaining_ar( src, dst, width, height, pad );
  }
  else if( option == "chip" || option == "scale" ||
           option == "chip_and_original" )
  {
    if( scale_factor == 1.0 )
    {
      dst = src;
    }
    else
    {
      cv::resize( src, dst, cv::Size(), scale_factor, scale_factor );
      scale = scale_factor;
    }
  }
  else
  {
    throw std::runtime_error( "Invalid resize option: " + option );
  }

  return scale;
}

} } } // end namespace
