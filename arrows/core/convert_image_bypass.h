/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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
 * \brief Header defining the convert_image algorithm that acts as a bypass
 */

#ifndef KWIVER_ARROWS_CORE_CONVERT_IMAGE_BYPASS_H_
#define KWIVER_ARROWS_CORE_CONVERT_IMAGE_BYPASS_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/convert_image.h>

namespace kwiver {
namespace arrows {
namespace core {

/// A class for bypassing image conversion
class KWIVER_ALGO_CORE_EXPORT convert_image_bypass
  : public vital::algorithm_impl<convert_image_bypass, vital::algo::convert_image>
{
public:
  /// Name of the algorithm
  static constexpr char const* name = "bypass";

  /// Description of the algorithm
  static constexpr char const* description =
    "Performs no conversion and returns the given image container.";

   /// Default Constructor
  convert_image_bypass();

  /// Default image converter ( does nothing )
  /**
   * \param [in] img image to be converted
   * \returns the input image
   */
  virtual vital::image_container_sptr convert(vital::image_container_sptr img) const;
};

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_CORE_CONVERT_IMAGE_BYPASS_H_
