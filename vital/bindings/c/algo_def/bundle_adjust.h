// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::algo::bundle_adjust interface
 */

#ifndef VITAL_C_ALGO_BUNDLE_ADJUST_H_
#define VITAL_C_ALGO_BUNDLE_ADJUST_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>

#include <vital/bindings/c/types/camera_map.h>
#include <vital/bindings/c/types/landmark_map.h>
#include <vital/bindings/c/types/track_set.h>

/// Common algorithm functions
DECLARE_COMMON_ALGO_API( bundle_adjust )

/// Optimize the camera and landmark parameters given a set of tracks
/**
 * \param [in] algo bundle adjust algorithm instance
 * \param [in,out] cameras the cameras to optimize
 * \param [in,out] landmarks the landmarks to optimize
 * \param [in] tracks the tracks to use as constraints
 * \param [in] eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_algorithm_bundle_adjust_optimize( vital_algorithm_t *algo,
                                        vital_camera_map_t **cmap,
                                        vital_landmark_map_t **lmap,
                                        vital_trackset_t *tset,
                                        vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_BUNDLE_ADJUST_H_
