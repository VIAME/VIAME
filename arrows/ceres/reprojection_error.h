// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for Ceres reprojection error functions
 */

#ifndef KWIVER_ARROWS_CERES_REPROJECTION_ERROR_H_
#define KWIVER_ARROWS_CERES_REPROJECTION_ERROR_H_

#include <vital/vital_config.h>
#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <arrows/ceres/types.h>

namespace kwiver {
namespace arrows {
namespace ceres {

/// Factory to create Ceres cost functions for each lens distortion type
KWIVER_ALGO_CERES_EXPORT
::ceres::CostFunction*
create_cost_func(LensDistortionType ldt, double x, double y);

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
