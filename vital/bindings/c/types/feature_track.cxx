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
 * \brief C Interface to vital::feature_track implementation
 */

#include "feature_track.h"

#include <vital/types/feature_track_set.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/descriptor.h>
#include <vital/bindings/c/helpers/feature.h>
#include <vital/bindings/c/helpers/track.h>


using namespace kwiver;

////////////////////////////////////////////////////////////////////////////////
// Track State

/// Create a new track state
vital_track_state_t*
vital_feature_track_state_new( int64_t frame, vital_feature_t *f,
                               vital_descriptor_t *d, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_new", eh,
    vital::feature_sptr f_sptr;
    vital::descriptor_sptr d_sptr;
    if( f ) f_sptr = vital_c::FEATURE_SPTR_CACHE.get( f );
    if( d ) d_sptr = vital_c::DESCRIPTOR_SPTR_CACHE.get( d );
    vital::track_state_data_sptr td_sptr(
      new vital::feature_track_state_data(f_sptr, d_sptr) );
    vital::track::track_state *ts =
      new vital::track::track_state(frame, td_sptr);
    return reinterpret_cast< vital_track_state_t* >( ts );
  );
  return 0;
}


/// Get a track state's feature
vital_feature_t*
vital_track_state_feature( vital_track_state_t *ts, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_feature", eh,
    REINTERP_TYPE( vital::track::track_state, ts, ts_ptr );
    // increase cross-boundary reference count if non-null
    if( dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->feature )
    {
      vital_c::FEATURE_SPTR_CACHE.store(
        dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->feature );
    }
    return reinterpret_cast< vital_feature_t* >(
      dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->feature.get() );
  );
  return 0;
}


/// Get a track state's descriptor
vital_descriptor_t*
vital_track_state_descriptor( vital_track_state_t *ts, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_descriptor", eh,
    REINTERP_TYPE( vital::track::track_state, ts, ts_ptr );
    // increase cross-boundary reference count if non-null
    if( dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->descriptor )
    {
      vital_c::DESCRIPTOR_SPTR_CACHE.store(
        dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->descriptor );
    }
    return reinterpret_cast< vital_descriptor_t* >(
      dynamic_cast< vital::feature_track_state_data* >( ts_ptr->data.get() )->descriptor.get() );
  );
  return 0;
}

