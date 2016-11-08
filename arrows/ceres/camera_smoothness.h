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
 * \brief Header for Ceres camera smoothness error functions for regularization
 */

#ifndef KWIVER_ARROWS_CERES_CAMERA_SMOOTHNESS_H_
#define KWIVER_ARROWS_CERES_CAMERA_SMOOTHNESS_H_


#include <vital/vital_config.h>
#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <arrows/ceres/types.h>

#include <ceres/ceres.h>

namespace kwiver {
namespace arrows {
namespace ceres {


/// Ceres camera smoothness functor
class camera_position_smoothness
{
public:
  /// Constructor
  camera_position_smoothness(const double smoothness)
      : smoothness_(smoothness) {}

  /// Position smoothness error functor for use in Ceres
  /**
   * \param [in] prev_pos: Camera pose data block at previous time
   * \param [in] curr_pos: Camera pose data block at current time
   * \param [in] next_pos: Camera pose data block at next time
   * \param [out] residuals
   *
   * Camera pose blocks contain 6 parameters:
   *   3 for rotation(angle axis), 3 for center
   * Only the camera centers are used in this function to penalize
   * the difference between current position and the average between
   * previous and next positions.
   */
  template <typename T> bool operator()(const T* const prev_pose,
                                        const T* const curr_pose,
                                        const T* const next_pose,
                                        T* residuals) const
  {
    residuals[0] = smoothness_ * (prev_pose[3] + next_pose[3] - T(2) * curr_pose[3]);
    residuals[1] = smoothness_ * (prev_pose[4] + next_pose[4] - T(2) * curr_pose[4]);
    residuals[2] = smoothness_ * (prev_pose[5] + next_pose[5] - T(2) * curr_pose[5]);

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(const double s)
  {
    typedef camera_position_smoothness Self;
    return new ::ceres::AutoDiffCostFunction<Self, 3, 6, 6, 6>(new Self(s));
  }

  double smoothness_;
};


/// Factory to create Ceres cost functions for camera path smoothness
::ceres::CostFunction*
create_smoothness_cost_func(const double smoothness)
{
  return camera_position_smoothness::create(smoothness);
}

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif // KWIVER_ARROWS_CERES_CAMERA_SMOOTHNESS_H_
