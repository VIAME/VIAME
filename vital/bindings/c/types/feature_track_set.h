/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief C Interface to vital::feature_track definition
 */

#ifndef VITAL_C_FEATURE_TRACK_SET_H_
#define VITAL_C_FEATURE_TRACK_SET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "track.h"

#include <vital/bindings/c/types/descriptor.h>
#include <vital/bindings/c/types/feature.h>


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
