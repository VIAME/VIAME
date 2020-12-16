// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
