// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV camera intrinsics.  Converts kwiver intrinsics to OpenCV intrinsics.
 */

#ifndef KWIVER_ARROWS_OCV_CAMERA_INTRINSICS_H_
#define KWIVER_ARROWS_OCV_CAMERA_INTRINSICS_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vector>
#include <vital/vital_config.h>

#include <vital/types/camera_intrinsics.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// return OpenCV distortion coefficients given the camera intrinsics
KWIVER_ALGO_OCV_EXPORT
std::vector<double>
get_ocv_dist_coeffs(vital::camera_intrinsics_sptr intrinsics);

/// return OpenCV formatted distortion coefficients based on vital distortion coefficients
KWIVER_ALGO_OCV_EXPORT
std::vector<double>
dist_coeffs_to_ocv(std::vector<double> const& vital_dist_coeffs);

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
