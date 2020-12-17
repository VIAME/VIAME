// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::landmark_map
 */

#ifndef VITAL_C_LANDMARK_MAP_H_
#define VITAL_C_LANDMARK_MAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/landmark.h>
#include <vital/bindings/c/vital_c_export.h>

/// vital::landmark_map Opaque structure
typedef struct vital_landmark_map_s vital_landmark_map_t;

/// Create a new simple landmark map from an array of landmarks
/**
 * \param landmarks Array of landmark instances to store in this map. Parallel
 *                  to \c lm_ids.
 * \param lm_ids Array of landmark IDs for each landmark. Parallel to
 *               \c landmarks array.
 * \param length The size of the \c landmarks and \c lm_ids arrays.
 * \param eh Vital error handle instance
 * \returns New landmark map instance
 */
VITAL_C_EXPORT
vital_landmark_map_t*
vital_landmark_map_new( vital_landmark_t const **landmarks,
                        int64_t const *lm_ids,
                        size_t length,
                        vital_error_handle_t *eh );

/// Create a new, empty landmark map
/**
 * \param eh Vital error handle instance
 * \returns New, empty landmark map instance
 */
VITAL_C_EXPORT
vital_landmark_map_t*
vital_landmark_map_new_empty( vital_error_handle_t *eh );

/// Destroy a landmark map instance
/**
 * \param lm Landmark map instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_landmark_map_destroy( vital_landmark_map_t *lm, vital_error_handle_t *eh);

/// Get the size of the landmark map
/**
 * \param lm Landmark map instance
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
size_t
vital_landmark_map_size( vital_landmark_map_t const *lm,
                         vital_error_handle_t *eh );

/// Get the landmarks contained in this map
/**
 * Number of landmarks and IDs returned is equal to the return of
 * \c vital_landmark_map_size for the given landmark map instance.
 *
 * \param lm Landmark map instance to get landmarks from
 * \param[out] length The size of the output arrays
 * \param[out] lm_ids Pointer that will be set to a new array of landmark Ids.
 *             This is parallel to \c landmarks.
 * \param[out] landmarks Pointer that will be set to a new array of vital
 *             landmark instances contained in the map. This is parallel to
 *             \c lm_ids.
 * \param eh Vital error handle instance
 * \returns New array of vital landmark instances contained in the map. This is
 *          parallel to the IDs output through \c lm_ids.
 */
VITAL_C_EXPORT
void
vital_landmark_map_landmarks( vital_landmark_map_t const *lm,
                              int64_t **lm_ids,
                              vital_landmark_t ***landmarks,
                              vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_LANDMARK_MAP_H_
