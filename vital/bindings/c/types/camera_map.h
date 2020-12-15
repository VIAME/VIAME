// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
