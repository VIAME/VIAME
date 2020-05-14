/*ckwg +29
 * Copyright 2016, 2019 by Kitware, Inc.
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

#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <arrows/ceres/types.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace kwiver {
namespace arrows {
namespace ceres {


/// Ceres camera smoothness functor
class camera_position_smoothness
{
public:
  /// Constructor
  camera_position_smoothness(const double smoothness,
                             const double fraction = 0.5)
      : smoothness_(smoothness),
        f1_(1.0-fraction),
        f2_(fraction)
  {}

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
    residuals[0] = smoothness_ *
      (f1_ * prev_pose[3] + f2_ * next_pose[3] - curr_pose[3]);
    residuals[1] = smoothness_ *
      (f1_ * prev_pose[4] + f2_ * next_pose[4] - curr_pose[4]);
    residuals[2] = smoothness_ *
      (f1_ * prev_pose[5] + f2_ * next_pose[5] - curr_pose[5]);

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(const double s, const double f = 0.5)
  {
    typedef camera_position_smoothness Self;
    return new ::ceres::AutoDiffCostFunction<Self, 3, 6, 6, 6>(new Self(s, f));
  }

  double smoothness_;
  double f1_;
  double f2_;
};


/// Ceres camera limit forward motion functor
/**
 *  This class is to reglarize camera motion to minimize the amount of motion
 *  in the camera looking direction.  This is useful with zoom lenses at long
 *  focal lengths where distance and zoom are ambiguous.  Adding this
 *  constraint allows the optimization to prefer fast zoom changes over fast
 *  position change.
 */
class camera_limit_forward_motion
{
public:
  /// Constructor
  camera_limit_forward_motion(const double scale)
      : scale_(scale) {}

  /// Camera forward motion error functor for use in Ceres
  /**
   * \param [in] pose1: Camera pose data block at time 1
   * \param [in] pose2: Camera pose data block at time 2
   * \param [out] residuals
   *
   * Camera pose blocks contain 6 parameters:
   *   3 for rotation(angle axis), 3 for center
   */
  template <typename T> bool operator()(const T* const pose1,
                                        const T* const pose2,
                                        T* residuals) const
  {
    // Apply external parameters (Pose)
    const T* rotation1 = pose1;
    const T* rotation2 = pose2;
    const T* center1 = pose1 + 3;
    const T* center2 = pose2 + 3;

    T baseline[3];
    baseline[0] = center2[0] - center1[0];
    baseline[1] = center2[1] - center1[1];
    baseline[2] = center2[2] - center1[2];

    T rotated_baseline1[3];
    T rotated_baseline2[3];
    // Rotate the point according the camera rotation
    ::ceres::AngleAxisRotatePoint(rotation1,
                                  baseline,
                                  rotated_baseline1);
    ::ceres::AngleAxisRotatePoint(rotation2,
                                  baseline,
                                  rotated_baseline2);


    residuals[0] = scale_ * rotated_baseline1[2] *
                   scale_ * rotated_baseline2[2];

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(const double s)
  {
    typedef camera_limit_forward_motion Self;
    return new ::ceres::AutoDiffCostFunction<Self, 1, 6, 6>(new Self(s));
  }

  /// the magnitude of this constraint
  double scale_;
};


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
