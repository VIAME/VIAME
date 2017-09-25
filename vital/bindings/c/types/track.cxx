/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * \brief C Interface to vital::track implementation
 */

#include "track.h"

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/track.h>


namespace kwiver {
namespace vital_c {

/// Cache for saving shared pointer references for pointers in use
SharedPointerCache< vital::track, vital_track_t >
  TRACK_SPTR_CACHE( "track" );

SharedPointerCache< vital::track_state, vital_track_state_t >
  TRACK_STATE_SPTR_CACHE( "track_state" );
} }


using namespace kwiver;

////////////////////////////////////////////////////////////////////////////////
// Track State


/// Create a new track state
vital_track_state_t*
vital_track_state_new( int64_t frame, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_new", eh,
    vital::track_state_sptr ts_sptr( new vital::track_state( frame ) );
    kwiver::vital_c::TRACK_STATE_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast< vital_track_state_t* >( ts_sptr.get() );
  );
  return 0;
}


/// Destroy a track state instance
void
vital_track_state_destroy( vital_track_state_t *ts, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_destroy", eh,
    kwiver::vital_c::TRACK_STATE_SPTR_CACHE.erase( ts );
  );
}


/// Get a track state's frame ID
int64_t
vital_track_state_frame_id( vital_track_state_t *ts, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_state_frame_id", eh,
    REINTERP_TYPE( vital::track_state, ts, ts_ptr );
    return ts_ptr->frame();
  );
  return 0;
}


////////////////////////////////////////////////////////////////////////////////
// Track

/// Create a new track
vital_track_t*
vital_track_new( vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_new", eh,
    kwiver::vital::track_sptr t_sptr = kwiver::vital::track::create();
    kwiver::vital_c::TRACK_SPTR_CACHE.store( t_sptr );
    return reinterpret_cast<vital_track_t*>( t_sptr.get() );
  );
  return 0;
}


/// Destroy a VITAL track pointer
void
vital_track_destroy( vital_track_t *track,
                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track::destroy", eh,
    kwiver::vital_c::TRACK_SPTR_CACHE.erase( track );
  );
}


/// Get the ID of the track
int64_t
vital_track_id( vital_track_t const *t, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_id", eh,
    return vital_c::TRACK_SPTR_CACHE.get( t )->id();
  );
  return 0;
}


/// Set the track identification number
void
vital_track_set_id( vital_track_t *t, int64_t i, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_set_id", eh,
    vital_c::TRACK_SPTR_CACHE.get( t )->set_id( i );
  );
}


/// Access the first frame number covered by this track
int64_t
vital_track_first_frame( vital_track_t const *t, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_first_frame", eh,
    return vital_c::TRACK_SPTR_CACHE.get( t )->first_frame();
  );
  return 0;
}


/// Access the last frame number covered by this track
int64_t
vital_track_last_frame( vital_track_t const *t, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_last_frame", eh,
    return vital_c::TRACK_SPTR_CACHE.get( t )->last_frame();
  );
  return 0;
}


/// Return the set of all frame IDs covered by this track
int64_t*
vital_track_all_frame_ids( vital_track_t const *t, size_t *n,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_all_frame_ids", eh,
    auto t_sptr = vital_c::TRACK_SPTR_CACHE.get( t );
    std::set<int64_t> s = t_sptr->all_frame_ids();
    *n = s.size();
    int64_t *r = (int64_t*)malloc(sizeof(int64_t) * s.size());
    size_t i = 0;
    for( std::set<int64_t>::iterator it = s.begin();
         it != s.end();
         ++it, ++i )
    {
      r[i] = *it;
    }
    return r;
  );
  return NULL;
}


/// Get the number of states in the track
size_t
vital_track_size( vital_track_t const *track,
                  vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track::size", eh,
    return vital_c::TRACK_SPTR_CACHE.get( track )->size();
  );
  return 0;
}


/// Return whether or not this track has any states
VITAL_C_EXPORT
bool
vital_track_empty( vital_track_t const *track,
                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track::empty", eh,
    return kwiver::vital_c::TRACK_SPTR_CACHE.get( track )->empty();
  );
  return true;
}


/// Append a track state to this track
bool
vital_track_append_state( vital_track_t *t, vital_track_state_t *ts,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_append_state", eh,
    auto t_sptr = vital_c::TRACK_SPTR_CACHE.get( t );
    auto ts_sptr = vital_c::TRACK_STATE_SPTR_CACHE.get( ts );
    if( t_sptr && ts_sptr )
    {
      return t_sptr->append( ts_sptr );
    }
  );
  return false;
}


/// Find the track state matching the given frame ID
vital_track_state_t*
vital_track_find_state( vital_track_t *t, int64_t frame,
                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_track_find_state", eh,
    auto t_sptr = vital_c::TRACK_SPTR_CACHE.get( t );
    auto it = t_sptr->find( frame );
    if( it != t_sptr->end() )
    {
      vital::track_state_sptr const &ts = *it;

      // Store this state in our cache and return a c_ptr to it. The state already
      // exists so it doesn't need to be re-created with new, and when this function's
      // ptr is returned, a new object will be created using the default python c_ptr
      // VitalObject constructor, that will handle deleting the state sptr from the
      // cache when that python object goes out of scope.
      kwiver::vital_c::TRACK_STATE_SPTR_CACHE.store( ts );
      return reinterpret_cast< vital_track_state_t* >( ts.get() );
    }
  );
  return NULL;
}
