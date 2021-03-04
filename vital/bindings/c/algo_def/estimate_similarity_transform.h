// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::algo::estimate_similarity_transform interface
 */

#ifndef VITAL_C_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
#define VITAL_C_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_

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

DECLARE_COMMON_ALGO_API( estimate_similarity_transform )

/// Estimate the similarity transform between two corresponding point sets
/**
 * This function can fail when from and to point sets are misaligned,
 * insufficient or degenerate, setting an error code of 1.
 *
 * \param algo The algorithm instance
 * \param n    Length of \c from and \c to arrays
 * \param from List of length N of 3D points in the from space.
 * \param to   List of length N of 3D points in the to space.
 * \param eh   Vital error handle instance
 *
 * \returns An estimated similarity transform mapping 3D points in the
 *          \c from space to points in the \c to space.
 */
VITAL_C_EXPORT
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_transform_points(
  vital_algorithm_t const *algo,
  size_t n,
  vital_eigen_matrix3x1d_t const **from,
  vital_eigen_matrix3x1d_t const **to,
  vital_error_handle_t *eh
);

/// Estimate the similarity transform between two corresponding camera maps
/**
 * Cameras with corresponding frame IDs in the two maps are paired for
 * transform estimation. Cameras with no corresponding frame ID in the other
 * map are ignored. An exception is set if there are no shared frame IDs between
 * the two provided maps (nothing to pair).
 *
 * This function can fail when the from and to point sets are misaligned,
 * insufficient or degenerate, setting an error code of 1.
 *
 * \param algo The algorithm instance
 * \param from Map of original cameras, sharing N frames with the transformed
 *             cameras, where N > 0.
 * \param to   Map of transformed cameras, sharing N frames with the original
 *             cameras, where N > 0.
 * \param eh   Vital error handle instance
 *
 * \returns An estimated similarity transform mapping camera centers in the
 *          \c from space to camera centers in the \c to space.
 */
VITAL_C_EXPORT
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_camera_map(
  vital_algorithm_t const *algo,
  vital_camera_map_t const *from, vital_camera_map_t const *to,
  vital_error_handle_t *eh
);

/// Estimate the similarity transform between two corresponding landmark maps
/**
 * Landmarks with corresponding frame IDs in the two maps are paired for
 * transform estimation. Landmarks with no corresponding frame ID in the
 * other map are ignored. An exception is set if there are no
 * shared frame IDs between the two provided maps (nothing to pair).
 *
 * This function can fail when the from and to point sets are misaligned,
 * insufficient or degenerate, setting an error code of 1.
 *
 * \param algo Algorithm instance
 * \param from Map of original landmarks, sharing N frames with the
 *             transformed landmarks, where N > 0.
 * \param to   Map of transformed landmarks, sharing N frames with the
 *             original landmarks, where N > 0.
 * \param eh   Vital error handle instance.
 *
 * \returns An estimated similarity transform mapping landmark centers in the
 *          \c from space to camera centers in the \c to space.
 */
VITAL_C_EXPORT
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_landmark_map(
  vital_algorithm_t const *algo,
  vital_landmark_map_t const *from, vital_landmark_map_t const *to,
  vital_error_handle_t *eh
);

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
