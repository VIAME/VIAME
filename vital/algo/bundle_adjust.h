/*ckwg +29
 * Copyright 2014-2019 by Kitware, Inc.
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
 * \brief Header defining abstract \link kwiver::vital::algo::bundle_adjust
 *        bundle adjustment \endlink algorithm
 */

#ifndef VITAL_ALGO_BUNDLE_ADJUST_H_
#define VITAL_ALGO_BUNDLE_ADJUST_H_

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/camera_map.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/landmark_map.h>
#include <vital/types/sfm_constraints.h>

#include <functional>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for bundle adjustment using feature tracks
class VITAL_ALGO_EXPORT bundle_adjust
  : public kwiver::vital::algorithm_def<bundle_adjust>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "bundle_adjust"; }

  /// Optimize the camera and landmark parameters given a set of feature tracks
  /**
   * Implementations of this function should not modify the underlying objects
   * contained in the input structures. Output references should either be new
   * instances or the same as input.
   *
   * \param [in,out] cameras the cameras to optimize
   * \param [in,out] landmarks the landmarks to optimize
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in] metadata the frame metadata to use as constraints
   */
  virtual void
  optimize(kwiver::vital::camera_map_sptr& cameras,
           kwiver::vital::landmark_map_sptr& landmarks,
           kwiver::vital::feature_track_set_sptr tracks,
           kwiver::vital::sfm_constraints_sptr constraints = nullptr) const = 0;

  /// Optimize the camera and landmark parameters given a set of feature tracks
  /**
   * \param [in,out] cameras the cameras to optimize
   * \param [in,out] landmarks the landmarks to optimize
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in] fixed_cameras frame ids for cameras to be fixed in the optimization
   * \param [in] fixed_landmarks landmark ids for landmarks to be fixed in the optimization
   * \param [in] metadata the frame metadata to use as constraints
   */
  virtual void
  optimize(kwiver::vital::simple_camera_perspective_map &cameras,
           kwiver::vital::landmark_map::map_landmark_t &landmarks,
           vital::feature_track_set_sptr tracks,
           const std::set<vital::frame_id_t>& fixed_cameras,
           const std::set<vital::landmark_id_t>& fixed_landmarks,
           kwiver::vital::sfm_constraints_sptr constraints = nullptr) const;

  /// Typedef for the callback function signature
  typedef std::function<bool(kwiver::vital::camera_map_sptr,
                             kwiver::vital::landmark_map_sptr,
                             kwiver::vital::feature_track_set_changes_sptr)> callback_t;

  /// Set a callback function to report intermediate progress
  virtual void set_callback(callback_t cb);

protected:
  bundle_adjust();

  /// The callback function
  callback_t m_callback;
};


/// type definition for shared pointer to a bundle adjust algorithm
typedef std::shared_ptr<bundle_adjust> bundle_adjust_sptr;

} } } // end namespace

#endif // VITAL_ALGO_BUNDLE_ADJUST_H_
