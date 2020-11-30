// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for conversions between vital and vpgl cameras
 */

#ifndef KWIVER_ARROWS_VXL_CAMERA_H_
#define KWIVER_ARROWS_VXL_CAMERA_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/camera_perspective.h>

#include <vpgl/vpgl_perspective_camera.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// Construct a camera_sptr from a vpgl_perspective_camera
template <typename T>
KWIVER_ALGO_VXL_EXPORT
vital::camera_perspective_sptr vpgl_camera_to_vital(const vpgl_perspective_camera<T>& vcam);

/// Convert a vpgl_perspective_camera to a vital::camera_
template <typename T>
KWIVER_ALGO_VXL_EXPORT
void vpgl_camera_to_vital(const vpgl_perspective_camera<T>& vcam,
                          vital::simple_camera_perspective& mcam);

/// Convert a vital::camera_ to a vpgl_perspective_camera
template <typename T>
KWIVER_ALGO_VXL_EXPORT
void vital_to_vpgl_camera(const vital::camera_perspective& mcam,
                          vpgl_perspective_camera<T>& vcam);

/// Convert a vpgl_calibration_matrix to a vital::camera_intrinsics_
template <typename T>
KWIVER_ALGO_VXL_EXPORT
void vpgl_calibration_to_vital(const vpgl_calibration_matrix<T>& vcal,
                               vital::simple_camera_intrinsics& mcal);

/// Convert a vital::camera_intrinsics_ to a vpgl_calibration_matrix
template <typename T>
KWIVER_ALGO_VXL_EXPORT
void vital_to_vpgl_calibration(const vital::camera_intrinsics& mcal,
                               vpgl_calibration_matrix<T>& vcal);

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
