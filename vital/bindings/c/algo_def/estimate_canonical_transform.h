// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
