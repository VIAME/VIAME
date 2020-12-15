// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to vital::object_track definition
 */

#ifndef VITAL_C_OBJECT_TRACK_SET_H_
#define VITAL_C_OBJECT_TRACK_SET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "track.h"

#include <vital/bindings/c/types/detected_object.h>
#include <vital/bindings/c/types/track_set.h>

////////////////////////////////////////////////////////////////////////////////
// Object Track State

/// Create a new track state
/**
 * \param frame Frame the state intersects
 * \param d Detection instance associated with this state. May be null.
 * \param eh Vital error handle instance
 * \returns new instance of a track state
 */
VITAL_C_EXPORT
vital_track_state_t*
vital_object_track_state_new( int64_t frame,
                              int64_t time,
                              vital_detected_object_t *d,
                              vital_error_handle_t *eh );

/// Get a track state's object detection
/**
 * \param td Track state data instance
 * \param eh Vital error handle instance
 * \returns New reference to the object instance of the track state
 */
VITAL_C_EXPORT
vital_detected_object_t*
vital_object_track_state_detection( vital_track_state_t *td,
                                    vital_error_handle_t *eh );

////////////////////////////////////////////////////////////////////////////////
// Object Track Set

/// Create a new object track set from an array of track instances
/**
 * The given track array may be freed after calling this function as the
 * underlying instance references are shared into the track set.
 *
 * This may be given a length of 0 and a null pointer to create an empty track
 * set.
 *
 * \param length The number of tracks in the given array
 * \param tracks The array of tracks to create a set out of
 * \returns New track set instance containing the provided tracks
 */
VITAL_C_EXPORT
vital_trackset_t*
vital_object_trackset_new( size_t length, vital_track_t **tracks,
                           vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_OBJECT_TRACK_SET_H_
