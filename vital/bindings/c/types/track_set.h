/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
vital_trackset_new( size_t length, vital_track_t **tracks );


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
vital_trackset_size( vital_trackset_t *track_set,
                     vital_error_handle_t *eh );


/// Write track set to the given filepath
VITAL_C_EXPORT
void
vital_trackset_write_track_file( vital_trackset_t* ts,
                                 char const *filepath,
                                 vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_TRACK_SET_H_
