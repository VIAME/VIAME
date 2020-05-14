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
 * \brief Interface to abstract filter image algorithm
 */

#ifndef VITAL_ALGO_IMAGE_FILTER_H
#define VITAL_ALGO_IMAGE_FILTER_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for image set filter algorithms.
/**
 * This interface supports arrows/algorithms that do a pixel by pixel
 * image modification, such as image enhancement. The resultant image
 * must be the same size as the input image.
 */
class VITAL_ALGO_EXPORT image_filter
  : public kwiver::vital::algorithm_def<image_filter>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "image_filter"; }

  /// Filter a  input image and return resulting image
  /**
   * This method implements the filtering operation. The method does
   * not modify the image in place. The resulting image must be a
   * newly allocated image which is the same size as the input image.
   *
   * \param image_data Image to filter.
   * \returns a filtered version of the input image
   */
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data ) = 0;

protected:
  image_filter();

};

/// type definition for shared pointer to a image_filter algorithm
typedef std::shared_ptr<image_filter> image_filter_sptr;


} } } // end namespace

#endif /* VITAL_ALGO_IMAGE_FILTER_H */
