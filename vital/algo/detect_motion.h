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
 * \brief Interface to algorithms for motion detection
 */

#ifndef VITAL_ALGO_DETECT_MOTION_H
#define VITAL_ALGO_DETECT_MOTION_H


#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>


namespace kwiver {
namespace vital {
namespace algo {

/// \brief Abstract base class for motion detection algorithms.
class VITAL_ALGO_EXPORT detect_motion
  : public kwiver::vital::algorithm_def<detect_motion>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "detect_motion"; }

  /// Detect motion from a sequence of images
  /**
   * This method detects motion of foreground objects within a
   * sequence of images in which the background remains stationary.
   * Sequential images are passed one at a time. Motion estimates
   * are returned for each image as a heat map with higher values
   * indicating greater confidence.
   *
   * \param ts Timestamp for the input image
   * \param image Image from a sequence
   * \param reset_model Indicates that the background model should
   * be reset, for example, due to changes in lighting condition or
   * camera pose
   *
   * \returns A heat map image is returned indicating the confidence
   * that motion occurred at each pixel. Heat map image is single channel
   * and has the same width and height dimensions as the input image.
   */
  virtual image_container_sptr
    process_image( const timestamp& ts,
                   const image_container_sptr image,
                   bool reset_model ) = 0;

protected:
  detect_motion();

};

/// type definition for shared pointer to a detect_motion algorithm
typedef std::shared_ptr<detect_motion> detect_motion_sptr;

} } } // end namespace

#endif // VITAL_ALGO_DETECT_MOTION_H
