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
 * \brief Header for draw_detected_object_set
 */

#ifndef VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
#define VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class VITAL_ALGO_EXPORT draw_detected_object_set
  : public kwiver::vital::algorithm_def<draw_detected_object_set>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "draw_detected_object_set"; }

  /// Draw detected object boxes on Image.
  /**
   * This method draws the detections on a copy of the image. The
   * input image is unmodified. The actual boxes that are drawn are
   * controlled by the configuration for the implementation.
   *
   * @param detected_set Set of detected objects
   * @param image Boxes are drawn in this image
   *
   * @return Image with boxes and other annotations added.
   */
  virtual kwiver::vital::image_container_sptr
    draw( kwiver::vital::detected_object_set_sptr detected_set,
          kwiver::vital::image_container_sptr image ) = 0;

protected:
  draw_detected_object_set();

};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr<draw_detected_object_set> draw_detected_object_set_sptr;

} } } // end namespace

#endif // VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
