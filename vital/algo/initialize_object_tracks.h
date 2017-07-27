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
 * \brief initialize_object_tracks algorithm definition
 */

#ifndef VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_MATRIX_H_
#define VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_MATRIX_H_

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>

#include <vital/types/timestamp.h>
#include <vital/types/object_track_set.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for computing association cost matrices for tracking
class VITAL_ALGO_EXPORT initialize_object_tracks
  : public kwiver::vital::algorithm_def<initialize_object_tracks>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "initialize_object_tracks"; }

  /// Initialize new object tracks given detections.
  /**
   * \param ts frame ID
   * \param image contains the input image for the current frame
   * \param detections detected object sets from the current frame
   * \returns newly initialized tracks
   */
  virtual kwiver::vital::object_track_set_sptr
  initialize( kwiver::vital::timestamp ts,
              kwiver::vital::image_container_sptr image,
              kwiver::vital::detected_object_set_sptr detections ) const = 0;

protected:
  initialize_object_tracks();

};


/// Shared pointer for initialize_object_tracks algorithm definition class
typedef std::shared_ptr<initialize_object_tracks> initialize_object_tracks_sptr;


} } } // end namespace

#endif // VITAL_ALGO_INITIALIZE_OBJECT_TRACKS_H_
