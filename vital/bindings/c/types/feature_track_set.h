// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to vital::feature_track definition
 */

#ifndef VITAL_C_FEATURE_TRACK_SET_H_
#define VITAL_C_FEATURE_TRACK_SET_H_

#include "track.h"

#include <vital/bindings/c/types/descriptor.h>
#include <vital/bindings/c/types/feature.h>

#ifdef __cplusplus
extern "C"
{
#endif

////////////////////////////////////////////////////////////////////////////////
// Feature Track State

/// Create a new feature track state
/**
 * \param frame Frame ID for the state
 * \param f Feature instance associated with this state. May be null.
 * \param d Descriptor instance associated with this state. May be null.
 * \param eh Vital error handle instance
 * \returns new instance of a track state
 */
VITAL_C_EXPORT
vital_track_state_t*
vital_feature_track_state_new( int64_t frame,
                               vital_feature_t *f,
                               vital_descriptor_t *d,
                               vital_error_handle_t *eh );

/// Get a track state's feature
/**
 * \param td Track state instance
 * \param eh Vital error handle instance
 * \returns New reference to the Feature instance of the track state
 */
VITAL_C_EXPORT
vital_feature_t*
vital_feature_track_state_feature( vital_track_state_t *td,
                                   vital_error_handle_t *eh );

/// Get a track state's descriptor
/**
 * \param td Track state instance
 * \param eh Vital error handle instance
 * \returns New reference to the Descriptor instance of the track state
 */
VITAL_C_EXPORT
vital_descriptor_t*
vital_feature_track_state_descriptor( vital_track_state_t *td,
                                      vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_FEATURE_TRACK_SET_H_
