// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::algo::triangulate_landmarks interface
 */

#ifndef VITAL_C_ALGO_TRIANGULATE_LANDMARKS_H_
#define VITAL_C_ALGO_TRIANGULATE_LANDMARKS_H_

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

DECLARE_COMMON_ALGO_API( triangulate_landmarks )

/// Triangulate the landmark locations given sets of cameras and tracks
/**
 * This function only triangulates the landmarks with indices in the
 * landmark map and which have support in the tracks and cameras.
 *
 * \param[in] algo triangulate_landmarks algorithm instance
 * \param[in] cameras the cameras viewing the landmarks
 * \param[in] tracks the tracks to use as constraints
 * \param[in,out] landmarks the landmarks to triangulate
 * \param[in] eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_algorithm_triangulate_landmarks_triangulate( vital_algorithm_t *algo,
                                                   vital_camera_map_t *cameras,
                                                   vital_trackset_t *tracks,
                                                   vital_landmark_map_t **landmarks,
                                                   vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_TRIANGULATE_LANDMARKS_H_
