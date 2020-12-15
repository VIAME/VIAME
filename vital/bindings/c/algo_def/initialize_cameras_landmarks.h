// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
