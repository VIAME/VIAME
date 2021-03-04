// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to vital::track_set definition
 */

#ifndef VITAL_C_TRACK_SET_H_
#define VITAL_C_TRACK_SET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/track.h>

/// Opaque structure
typedef struct vital_trackset_s vital_trackset_t;

/// Create a new track set from an array of track instances
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
vital_trackset_new( size_t length, vital_track_t **tracks,
                    vital_error_handle_t *eh );

/// Adopt existing track set from sptr
/**
 *
 * \param sptr address of track set sptr
 *
 * \returns trackset opaque handle
 */
VITAL_C_EXPORT
vital_trackset_t*
vital_trackset_from_sptr( void* sptr );

/// Create a new track set as read from file
/**
 * NOTE: The simple track file format currently defined does not represent
 * all data within a track, but only the minimal subset of data needed for
 * running sparse bundle adjustment.
 *
 * This function may fail if:
 *  - The given filepath is not found
 *  - The given file cannot be read or parsed
 *  - Other filesystem errors
 *
 * \param filepath The path to the file to read in.
 * \param eh An error handle instance
 * \returns New track set instance containing the tracks defined in the given
 *          file.
 */
VITAL_C_EXPORT
vital_trackset_t*
vital_trackset_new_from_file( char const *filepath,
                              vital_error_handle_t *eh );

/// Destroy a track set instance
/**
 * This function destroys the referenced track set. The supplied
 * handle is removed from the internal cache as the associated smart
 * pointer. The actual track set data may be deleted if there are no
 * more ref counts.
 *
 * \param track_set opaque pointer to track set instance
 * \param eh Error handle instance.
 */
VITAL_C_EXPORT
void
vital_trackset_destroy( vital_trackset_t *track_set,
                        vital_error_handle_t *eh );

/// Get the size of the track set
VITAL_C_EXPORT
size_t
vital_trackset_size( vital_trackset_t const *track_set,
                     vital_error_handle_t *eh );

/// Write track set to the given filepath
VITAL_C_EXPORT
void
vital_trackset_write_track_file( vital_trackset_t const *ts,
                                 char const *filepath,
                                 vital_error_handle_t *eh );

/// Get array of contained track references
/**
 * The number of elements returned is equal to the size of this set.
 *
 * \param track_set Track set instance to extract all tracks from
 * \param eh Vital error handle instance
 * \returns New array of new track reference pointers
 */
VITAL_C_EXPORT
vital_track_t**
vital_trackset_tracks( vital_trackset_t const *track_set,
                       vital_error_handle_t *eh );

/// Get the set of all frame IDs covered by contained tracks
/**
 * \param trackset the track set instance
 * \param[out] length the number of frame IDs in the returned set
 * \param eh Vital error handle instance
 * \returns set of frame IDs as array
 */
VITAL_C_EXPORT
int64_t*
vital_trackset_all_frame_ids( vital_trackset_t const *trackset, size_t *length,
                              vital_error_handle_t *eh );

/// Get the set of all track IDs in the track set
/**
 * \param trackset the track set instance
 * \param[out] length the number of track IDs in the returned set
 * \param eh Vital error handle instance
 * \returns set of track IDs as array
 */
VITAL_C_EXPORT
int64_t*
vital_trackset_all_track_ids( vital_trackset_t const *trackset, size_t *length,
                              vital_error_handle_t *eh );

/// Get the first (smallest) frame number containing tracks
/**
 * \param trackset the track set instance
 * \param eh Vital error handle instance
 * \returns first frame ID covered by contained tracks
 */
VITAL_C_EXPORT
int64_t
vital_trackset_first_frame( vital_trackset_t const *trackset,
                            vital_error_handle_t *eh );

/// Get the last (largest) frame number containing tracks
/**
 * \param trackset the track set instance
 * \param eh Vital error handle instance
 * \returns last frame ID covered by contained tracks
 */
VITAL_C_EXPORT
int64_t
vital_trackset_last_frame( vital_trackset_t const *trackset,
                           vital_error_handle_t *eh );

/// Get the track in this set with the specified id
/**
 * A null pointer will be returned if the track cannot be found.
 *
 * \param trackset the track set instance
 * \param tid ID of the track to get
 * \param eh Vital error handle instance
 * \returns New reference to the contained track, or null if this set does not
 *          contain a track by the specified ID.
 */
VITAL_C_EXPORT
vital_track_t*
vital_trackset_get_track( vital_trackset_t const *trackset, int64_t tid,
                          vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_TRACK_SET_H_
