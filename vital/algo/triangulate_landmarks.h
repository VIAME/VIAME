/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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
 * \brief Header defining abstract \link kwiver::vital::algo::triangulate_landmarks
 *        triangulate landmarks \endlink algorithm
 */

#ifndef VITAL_ALGO_TRIANGULATE_LANDMARKS_H_
#define VITAL_ALGO_TRIANGULATE_LANDMARKS_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>


namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for triangulating landmarks
class VITAL_ALGO_EXPORT triangulate_landmarks
: public kwiver::vital::algorithm_def<triangulate_landmarks>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "triangulate_landmarks"; }

  /// Triangulate the landmark locations given sets of cameras and feature tracks
  /**
   * \param [in] cameras the cameras viewing the landmarks
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in,out] landmarks the landmarks to triangulate
   *
   * This function only triangulates the landmarks with indices in the
   * landmark map and which have support in the feature tracks and cameras
   */
  virtual void
  triangulate(kwiver::vital::camera_map_sptr cameras,
              kwiver::vital::feature_track_set_sptr tracks,
              kwiver::vital::landmark_map_sptr& landmarks) const = 0;

protected:
  triangulate_landmarks();

};


/// type definition for shared pointer to a triangulate landmarks algorithm
typedef std::shared_ptr<triangulate_landmarks> triangulate_landmarks_sptr;


} // end namespace algo
} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_TRIANGULATE_LANDMARKS_H_
