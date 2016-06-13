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
 * \brief C Interface to vital::track definition
 */

#ifndef VITAL_C_TRACK_H_
#define VITAL_C_TRACK_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>


/// Opaque structure for vital::track instances
typedef struct vital_track_s vital_track_t;


/// Create a new track
VITAL_C_EXPORT
vital_track_t*
vital_track_new();


/// Destroy a VITAL track pointer
VITAL_C_EXPORT
void
vital_track_destroy( vital_track_t *track,
                     vital_error_handle_t *eh );


/// Get the number of states in the track
VITAL_C_EXPORT
size_t
vital_track_size( vital_track_t *track,
                  vital_error_handle_t *eh );


/// Return whether or not this track has any states
VITAL_C_EXPORT
bool
vital_track_empty( vital_track_t *track,
                   vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif // VITAL_C_TRACK_H_
