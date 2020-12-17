// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to vital::track definition
 */

#ifndef VITAL_C_TRACK_H_
#define VITAL_C_TRACK_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>

/// Opaque structure for vital::track_state instances
typedef struct vital_track_state_s vital_track_state_t;
/// Opaque structure for vital::track instances
typedef struct vital_track_s vital_track_t;

////////////////////////////////////////////////////////////////////////////////

/// Create a new track state
/**
 * \param frame Frame the state intersects
 * \param eh Vital error handle instance
 * \returns new instance of a track state
 */
VITAL_C_EXPORT
vital_track_state_t*
vital_track_state_new( int64_t frame, vital_error_handle_t *eh );

/// Destroy a track state instance
/**
 * \param ts Track state instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_track_state_destroy( vital_track_state_t *ts, vital_error_handle_t *eh );

/// Get a track state's frame ID
/**
 * \param ts Track state instance
 * \param eh Vital error handle instance
 * \returns Frame ID of the track state
 */
VITAL_C_EXPORT
int64_t
vital_track_state_frame_id( vital_track_state_t *ts, vital_error_handle_t *eh );

////////////////////////////////////////////////////////////////////////////////
// Track

/// Create a new track
/**
 * \param eh Vital error handle instance
 * \returns New track instance
 */
VITAL_C_EXPORT
vital_track_t*
vital_track_new( vital_error_handle_t *eh );

/// Destroy a VITAL track pointer
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_track_destroy( vital_track_t *track,
                     vital_error_handle_t *eh );

/// Get the ID of the track
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 * \returns integer ID of the track
 */
VITAL_C_EXPORT
int64_t
vital_track_id( vital_track_t const *t, vital_error_handle_t *eh );

/// Set the track identification number
/**
 * \param track Track instance
 * \param i New ID to set.
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_track_set_id( vital_track_t *t, int64_t i, vital_error_handle_t *eh );

/// Access the first frame number covered by this track
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 * \returns ID of the first frame
 */
VITAL_C_EXPORT
int64_t
vital_track_first_frame( vital_track_t const *t, vital_error_handle_t *eh );

/// Access the last frame number covered by this track
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 * \returns ID of the last frame
 */
VITAL_C_EXPORT
int64_t
vital_track_last_frame( vital_track_t const *t, vital_error_handle_t *eh );

/// Return the set of all frame IDs covered by this track
/**
 * A null pointer is returned if there are no states in this track.
 *
 * \param track Track instance
 * \param eh Vital error handle instance
 * \param[out] n Number of unique frame IDs
 * \return Pointer to \c n unique frame IDs. Null if size is 0.
 */
VITAL_C_EXPORT
int64_t*
vital_track_all_frame_ids( vital_track_t const *t, size_t *n,
                           vital_error_handle_t *eh );

/// Access the track identification number
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
size_t
vital_track_size( vital_track_t const *track,
                  vital_error_handle_t *eh );

/// Return whether or not this track has any states
/**
 * \param track Track instance
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
bool
vital_track_empty( vital_track_t const *track,
                   vital_error_handle_t *eh );

/// Append a track state to this track
/**
 * The new track state must have a frame_id greater than the last frame in the
 * history. If such an append is attempted, nothing is added to this track.
 *
 * \param t Track instance to append to
 * \param ts Track state instance to append
 * \param eh Vital error handle instance
 * \returns True if successful, false if not correctly ordered.
 */
VITAL_C_EXPORT
bool
vital_track_append_state( vital_track_t *t, vital_track_state_t *ts,
                          vital_error_handle_t *eh );

/// Find the track state matching the given frame ID
/**
 * \param t the Track instance to search in
 * \param frame the frame ID to look for
 * \param eh Vital error handle instance
 * \returns New instance of the track state (shallow copy) at the given frame if
 *          there is one, or NULL if the frame ID is not represented in this
 *          track.
 */
VITAL_C_EXPORT
vital_track_state_t*
vital_track_find_state( vital_track_t *t, int64_t frame,
                        vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_TRACK_H_
