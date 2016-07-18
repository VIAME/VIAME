/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief C interface for vital::camera_map
 */

#ifndef VITAL_C_CAMERA_MAP_H_
#define VITAL_C_CAMERA_MAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

#include <vital/bindings/c/types/camera.h>
#include <vital/bindings/c/error_handle.h>


/// Opaque structure for vital::camera_map class
typedef struct vital_camera_map_s vital_camera_map_t;


/// New, simple camera map
/**
 * Given a two parallel arrays of frame number and cameras, create a new
 * camera map.
 *
 * If either array is NULL or if length is zero, the returned camera_map will
 * be empty.
 *
 * \param length The size of the parallel pointer arrays given that constitute
 *               the map.
 * \param frame_numbers[in] Pointer array of frame numbers of size \c length.
 *                          This should be parallel in association with the
 *                          \c cameras array.
 * \param cameras[in] Pointer array of camera instances. This should be parallel
 *                    in association with the \c frame_numbers array.
 * \param eh[in] Vital error handle instance.
 * \returns New instance of a camera map, storing the input relationship between
 *          frame numbers and cameras.
 */
VITAL_C_EXPORT
vital_camera_map_t* vital_camera_map_new( size_t length,
                                          int64_t *frame_numbers,
                                          vital_camera_t **cameras,
                                          vital_error_handle_t *eh );


/// Destroy the given camera_map
VITAL_C_EXPORT
void vital_camera_map_destroy( vital_camera_map_t *cam_map,
                               vital_error_handle_t *eh );


/// Return the number of cameras in the map
VITAL_C_EXPORT
size_t vital_camera_map_size( vital_camera_map_t *cam_map,
                              vital_error_handle_t *eh );


/// Set pointers to parallel arrays of frame numbers and camera instances
VITAL_C_EXPORT
void vital_camera_map_get_map( vital_camera_map_t *cam_map,
                               size_t *length,
                               int64_t **frame_numbers,
                               vital_camera_t ***cameras,
                               vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif // VITAL_C_CAMERA_MAP_H_
