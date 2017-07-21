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
 * \brief C Interface to vital::object_track implementation
 */

#include "object_track_set.h"

#include <vital/types/object_track_set.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/detected_object.h>
#include <vital/bindings/c/helpers/track.h>


using namespace kwiver;

////////////////////////////////////////////////////////////////////////////////
// Track State

/// Create a new track state
vital_track_state_t*
vital_object_track_state_new( int64_t frame, vital_detected_object_t *d,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_object_track_state_new", eh,
    vital::detected_object_sptr d_sptr;
    if( d ) d_sptr = vital_c::DOBJ_SPTR_CACHE.get( d );
    vital::track_state_data_sptr td_sptr(
      new vital::object_track_state_data( d_sptr ) );
    vital::track::track_state *ts =
      new vital::track::track_state( frame, td_sptr );
    return reinterpret_cast< vital_track_state_t* >( ts );
  );
  return 0;
}


/// Get a track state's object
vital_detected_object_t*
vital_track_state_object( vital_track_state_t *ts, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_object", eh,
    REINTERP_TYPE( vital::track::track_state, ts, ts_ptr );
    // increase cross-boundary reference count if non-null
    if( dynamic_cast< vital::object_track_state_data* >( ts_ptr->data.get() )->detection )
    {
      vital_c::DOBJ_SPTR_CACHE.store(
        dynamic_cast< vital::object_track_state_data* >( ts_ptr->data.get() )->detection );
    }
    return reinterpret_cast< vital_detected_object_t* >(
      dynamic_cast< vital::object_track_state_data* >( ts_ptr->data.get() )->detection.get() );
  );
  return 0;
}

