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
 * \brief vital::algo::initialize_cameras_landmarks interface
 */

#ifndef VITAL_C_ALGO_INITIALIZE_CAMERAS_LANDMARKS_H_
#define VITAL_C_ALGO_INITIALIZE_CAMERAS_LANDMARKS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/camera_map.h>
#include <vital/bindings/c/types/landmark_map.h>
#include <vital/bindings/c/types/track_set.h>
#include <vital/bindings/c/vital_c_export.h>


DECLARE_COMMON_ALGO_API( initialize_cameras_landmarks )


/// Initialize the camera and landmark parameters given a set of tracks
/**
 * \param[in] algo initialize cameras landmarks algorithm instance
 * \param[in,out] cameras Cameras to initialize
 * \param[in,out] landmarks Landmarks to initialize
 * \param[in] tracks Tracks to use as constraints
 * \param[in] eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_algorithm_initialize_cameras_landmarks_initialize( vital_algorithm_t *algo,
                                                         vital_camera_map_t **cameras,
                                                         vital_landmark_map_t **landmarks,
                                                         vital_trackset_t *tracks,
                                                         vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_INITIALIZE_CAMERAS_LANDMARKS_H_
