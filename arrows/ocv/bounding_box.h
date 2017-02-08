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
 * \brief Interface to bounding box utilities
 */

#ifndef ARROWS_OCV_BOUNDING_BOX_H
#define ARROWS_OCV_BOUNDING_BOX_H

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/types/bounding_box.h>
#ifndef KWIVER_HAS_OPENCV_VER_3
#include <opencv2/core/types_c.h>      // OCV rect
#else
#include <opencv2/core.hpp>            // OCV rect
#endif


namespace kwiver {
namespace arrows {
namespace ocv {

/**
 * @brief Convert CvRect to bounding_box
 *
 * This operator converts a CvRect to a kwiver bounding box.
 *
 * @param vbox CvRect to convert
 *
 * @return Equivalent bounding box.
 */
template <typename T>
kwiver::vital::bounding_box<T> convert( const CvRect& vbox )
{
  typename kwiver::vital::bounding_box<T>::vector_type bb_tl( vbox.x, vbox.y );
  return kwiver::vital::bounding_box<T>( bb_tl, vbox.width, vbox.height );
}


// ------------------------------------------------------------------
/**
 * @brief Convert bounding box to CvRect
 *
 * @param bbox Bounding box to convert
 *
 * @return Equivalent CvRect
 */
template <typename T>
CvRect convert(const kwiver::vital::bounding_box<T>& bbox )
{
  // Note that CvRect has integer values. If T is a floating type. the
  // fractions are turncated.
  return cvRect( static_cast<int>(bbox.min_x()),
                 static_cast<int>(bbox.min_y()),
                 static_cast<int>(bbox.width()),
                 static_cast<int>(bbox.height()));
}


} } } // end namespace

#endif /* ARROWS_OCV_BOUNDING_BOX_H */
