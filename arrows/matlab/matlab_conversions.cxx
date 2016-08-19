/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief Interface for MatLab conversion functions
 */

#include "matlab_conversions.h"

#include <matrix.h>  // matlab include
#include <arrows/ocv/image_container.h>
#include <opencv2/core/core.hpp>

#include <stdint.h>

namespace kwiver {
namespace arrows {
namespace matlab {

// ------------------------------------------------------------------
mxArraySptr
convert_to_mx_image( const kwiver::vital::image_container_sptr image )
{
  const size_t rows = image->height();
  const size_t cols = image->width();

  mxArraySptr mx_image = create_mxByteArray( rows, cols );
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image->get_image() );

  // Copy the pixels
  uint8_t* mx_mem = static_cast< uint8_t* > ( mxGetData( mx_image.get() ) );

  // convert from column major to row major.
  for ( size_t i = 0; i < rows; i++ )
  {
    for ( size_t j = 0; j < cols; j++ )
    {
      // row major indexing
      mx_mem[i * ( cols ) + j] = src.at< uint8_t > ( i, j );
    }
  }

  return mx_image;
}


} } } // end namespace
