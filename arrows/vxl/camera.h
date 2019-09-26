/*ckwg +29
 * Copyright 2014-2016, 2019 by Kitware, Inc.
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
 * \brief Header file for conversions between vital and vpgl cameras
 */

#ifndef KWIVER_ARROWS_VXL_CAMERA_H_
#define KWIVER_ARROWS_VXL_CAMERA_H_


#include <vital/vital_config.h>
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
