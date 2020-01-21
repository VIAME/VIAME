/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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
