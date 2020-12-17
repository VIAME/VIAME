// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to track_features algorithm definition
 */

#ifndef VITAL_C_ALGO_DEF_TRACK_FEATURES_H_
#define VITAL_C_ALGO_DEF_TRACK_FEATURES_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/image_container.h>
#include <vital/bindings/c/types/track_set.h>

/// Common algorithm definition API
DECLARE_COMMON_ALGO_API( track_features );

/// New track set of extended tracks using the current frame
VITAL_C_EXPORT
vital_trackset_t*
vital_algorithm_track_features_track( vital_algorithm_t *algo,
                                      vital_trackset_t *prev_tracks,
                                      unsigned int frame_num,
                                      vital_image_container_t *ic,
                                      vital_error_handle_t *eh );

/// New track set of extended tracks using the current frame, masked version
VITAL_C_EXPORT
vital_trackset_t*
vital_algorithm_track_features_track_with_mask( vital_algorithm_t *algo,
                                                vital_trackset_t *prev_tracks,
                                                unsigned int frame_num,
                                                vital_image_container_t *ic,
                                                vital_image_container_t *mask,
                                                vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_ALGO_DEF_TRACK_FEATURES_H_
