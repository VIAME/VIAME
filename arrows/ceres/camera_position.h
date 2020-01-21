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

#ifndef KWIVER_ARROWS_CERES_CAMERA_POSITION_H_
#define KWIVER_ARROWS_CERES_CAMERA_POSITION_H_

#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <arrows/ceres/types.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace kwiver {
namespace arrows {
namespace ceres {


/// Ceres camera smoothness functor
class camera_position
{
public:
  /// Constructor
  camera_position(vital::vector_3d const& position_prior_local)
      : position_prior_local_(position_prior_local) {}

  /// Position smoothness error functor for use in Ceres
  /**
   * \param [in] pose: Camera pose data block
   * \param [out] residuals
   *
   * Camera pose blocks contain 6 parameters:
   *   3 for rotation(angle axis), 3 for center
   * Only the camera center is used in this function to penalize
   * the difference between the position and the prior from metadata
   */
  template <typename T> bool operator()(const T* const pose,
                                        T* residuals) const
  {
    residuals[0] = T(position_prior_local_[0]) - pose[3];
    residuals[1] = T(position_prior_local_[1]) - pose[4];
    residuals[2] = T(position_prior_local_[2]) - pose[5];

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(vital::vector_3d const& position_prior_local )
  {
    typedef camera_position Self;
    return new ::ceres::AutoDiffCostFunction<Self, 3, 6>(new Self(position_prior_local));
  }

  vital::vector_3d position_prior_local_;
};


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
