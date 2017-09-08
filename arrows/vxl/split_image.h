/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Header for VXL split_image algorithm
 */

#ifndef KWIVER_ARROWS_VXL_SPLIT_IMAGE_H_
#define KWIVER_ARROWS_VXL_SPLIT_IMAGE_H_


#include <vital/vital_config.h>
#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/split_image.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A class for drawing various information about feature tracks
class KWIVER_ALGO_VXL_EXPORT split_image
: public vital::algorithm_impl<split_image, vital::algo::split_image>
{
public:

  /// Constructor
  split_image();

  /// Destructor
  virtual ~split_image();

  /// Split image
  virtual std::vector< kwiver::vital::image_container_sptr >
  split(kwiver::vital::image_container_sptr img) const;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_VXL_SPLIT_IMAGE_H_
