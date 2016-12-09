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
 * \brief File description here.
 */

#ifndef VITAL_C_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
#define VITAL_C_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/camera_map.h>
#include <vital/bindings/c/types/landmark_map.h>
#include <vital/bindings/c/types/similarity.h>
#include <vital/bindings/c/vital_c_export.h>


DECLARE_COMMON_ALGO_API( estimate_canonical_transform )


/// Estimate a canonical similarity transform for cameras and points
/**
 *
 *
 * \note This algorithm does not apply the transformation, it only estimates it.
 *
 * This function can fail when the is insufficient or degenerate, setting an
 * error code of 1.
 *
 * \param algo Algorithm instance
 * \param cam_map The camera map containing all the cameras
 * \param lm_map The landmark map containing all the 3D landmarks
 * \returns New estimated similarity transformation mapping the data to the
 *          canonical space.
 */
VITAL_C_EXPORT
vital_similarity_d_t*
vital_algorithm_estimate_canonical_transform_estimate( vital_algorithm_t *algo,
                                                       vital_camera_map_t const *cam_map,
                                                       vital_landmark_map_t const *lm_map,
                                                       vital_error_handle_t *eh );



#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
