// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for Ceres camera intrinsic prior cost functions
 */

#ifndef KWIVER_ARROWS_CERES_CAMERA_INTRINSIC_PRIOR_H_
#define KWIVER_ARROWS_CERES_CAMERA_INTRINSIC_PRIOR_H_

#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <arrows/ceres/types.h>

#include <ceres/ceres.h>

namespace kwiver {
namespace arrows {
namespace ceres {

/// Ceres camera smoothness functor
class camera_intrinsic_prior
{
public:
  /// Constructor
  camera_intrinsic_prior(double max_focal_len)
      : max_focal_len_(max_focal_len) {}

  /// Intrinsic prior error functor for use in Ceres
  /**
   * \param [in] int_parms: Camera intrinisics parameter block
   * \param [out] residuals
   */
  template <typename T> bool operator()(const T* const int_param,
                                        T* residuals) const
  {
    // parameter 0 is focal length
    residuals[0] = int_param[0] - T(max_focal_len_);
    // this is a one-sided cost that is only incurred when the focal length
    // exceeds the maximum value
    if (residuals[0] < T(0))
    {
      residuals[0] = T(0);
    }

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(double max_focal_len,
                                       std::size_t num_intrin_params)
  {
    using Self = camera_intrinsic_prior;
#define DO_CASE(N) \
  case N: \
    return new ::ceres::AutoDiffCostFunction<Self, 1, N>(new Self(max_focal_len))

    switch (num_intrin_params)
    {
      DO_CASE(5);
      DO_CASE(7);
      DO_CASE(10);
      DO_CASE(13);
    default:
      throw kwiver::vital::invalid_value("Unsupport camera intrinsics size, " +
        std::to_string(num_intrin_params) + ", in camera_intrinsic_prior");
      break;
    }
    return nullptr;
  }
#undef DO_CASE

  double max_focal_len_;
};

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
