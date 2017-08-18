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
 * \brief C Interface to kwiver::vital::track_set implementation
 */

#include "track_set.h"
#include "track_set.hxx"

#include <vector>

#include <vital/types/track_set.h>
#include <vital/io/track_set_io.h>
#include <vital/vital_foreach.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/track.h>
#include <vital/bindings/c/helpers/track_set.h>

namespace kwiver {
namespace vital_c {

SharedPointerCache< kwiver::vital::track_set, vital_trackset_t >
TRACK_SET_SPTR_CACHE( "track_set" );

}
}


using namespace kwiver;


/// Create a new track set from an array of track instances
vital_trackset_t*
vital_trackset_new( size_t length, vital_track_t **tracks,
                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_new", eh,
    std::vector< kwiver::vital::track_sptr > track_vec;
    for ( size_t i = 0; i < length; ++i )
    {
      track_vec.push_back( vital_c::TRACK_SPTR_CACHE.get( tracks[i] ) );
    }

    kwiver::vital::track_set_sptr ts_sptr( new kwiver::vital::track_set( track_vec )
                                         );

    kwiver::vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast< vital_trackset_t* > ( ts_sptr.get() );
    );
  return 0;
}


/// Adopt existing track set from sptr
vital_trackset_t*
vital_trackset_from_sptr( void* sptr )
{
  STANDARD_CATCH(
    "vital_trackset_from_sptr", NULL,

    kwiver::vital::track_set_sptr ts_sptr = *reinterpret_cast< kwiver::vital::track_set_sptr* >(sptr);
    kwiver::vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast<vital_trackset_t*>( ts_sptr.get() );
  );
  return 0;
}


/// Create a new track set as read from file
vital_trackset_t*
vital_trackset_new_from_file( char const *filepath,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_new_from_file", eh,
    vital::track_set_sptr ts_sptr( vital::read_track_file( filepath ) );
    vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast<vital_trackset_t*>( ts_sptr.get() );
  );
  return 0;
}


/// Destroy a track set instance
void
vital_trackset_destroy( vital_trackset_t *track_set,
                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_destroy", eh,
    vital_c::TRACK_SET_SPTR_CACHE.erase( track_set );
  );
}


/// Get the size of the track set
size_t
vital_trackset_size( vital_trackset_t const *track_set,
                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_size", eh,
    return vital_c::TRACK_SET_SPTR_CACHE.get( track_set )->size();
  );
  return 0;
}


/// Write track set to the given filepath
void
vital_trackset_write_track_file( vital_trackset_t const *ts,
                                 char const *filepath,
                                 vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_write_track_file", eh,
    vital::write_track_file(
      vital_c::TRACK_SET_SPTR_CACHE.get( ts ),
      filepath
    );
  );
}


/// Get array of contained track references
vital_track_t**
vital_trackset_tracks( vital_trackset_t const *track_set,
                       vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_tracks", eh,
    auto ts_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( track_set );
    auto tracks_v = ts_sptr->tracks();
    vital_track_t **track_array =
      (vital_track_t**)malloc( sizeof(vital_track_t*) * tracks_v.size() );
    for( size_t i=0; i < tracks_v.size(); ++i )
    {
      vital_c::TRACK_SPTR_CACHE.store( tracks_v[i] );
      track_array[i] = reinterpret_cast< vital_track_t* >( tracks_v[i].get() );
    }
    return track_array;
  );
  return 0;
}


/// Get the set of all frame IDs covered by contained tracks
int64_t*
vital_trackset_all_frame_ids( vital_trackset_t const *trackset, size_t *length,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_all_frame_ids", eh,
    auto ts_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( trackset );
    auto fid_set = ts_sptr->all_frame_ids();
    *length = fid_set.size();
    int64_t *frame_ids = (int64_t*)malloc(sizeof(int64_t) * (*length));
    size_t i=0;
    VITAL_FOREACH( int64_t const &fid, fid_set )
    {
      frame_ids[i] = fid;
      ++i;
    }
    return frame_ids;
  );
  return 0;
}


/// Get the set of all track IDs in the track set
int64_t*
vital_trackset_all_track_ids( vital_trackset_t const *trackset, size_t *length,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_all_track_ids", eh,
    auto ts_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( trackset );
    auto tid_set = ts_sptr->all_track_ids();
    *length = tid_set.size();
    int64_t *track_ids = (int64_t*)malloc( sizeof(int64_t) * (*length));
    size_t i = 0;
    VITAL_FOREACH( int64_t const &tid, tid_set )
    {
      track_ids[i] = tid;
      ++i;
    }
    return track_ids;
  );
  return 0;
}


/// Get the first (smallest) frame number containing tracks
int64_t
vital_trackset_first_frame( vital_trackset_t const *trackset,
                            vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_first_frame", eh,
    return vital_c::TRACK_SET_SPTR_CACHE.get( trackset )->first_frame();
  );
  return 0;
}


/// Get the last (largest) frame number containing tracks
int64_t
vital_trackset_last_frame( vital_trackset_t const *trackset,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_last_frame", eh,
    return vital_c::TRACK_SET_SPTR_CACHE.get( trackset )->last_frame();
  );
  return 0;
}


/// Get the track in this set with the specified id
vital_track_t*
vital_trackset_get_track( vital_trackset_t const *trackset, int64_t tid,
                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_trackset_get_track", eh,
    auto ts_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( trackset );
    auto t_sptr = ts_sptr->get_track( tid );
    // Cache track sptr if non-null
    if( t_sptr ) vital_c::TRACK_SPTR_CACHE.store( t_sptr );
    return reinterpret_cast< vital_track_t* >( t_sptr.get() );
  );
  return 0;
}


/// Create a vital_track_set_t around an existing shared pointer.
vital_trackset_t*
vital_track_set_new_from_sptr( kwiver::vital::track_set_sptr ts_sptr,
                               vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "vital_track_set_new_from_sptr", eh,
    // Store the shared pointer in our cache and return the handle.
    vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast< vital_trackset_t* >( ts_sptr.get() );
  );
  return NULL;
}


/// Get the vital::track_set shared pointer for a handle.
kwiver::vital::track_set_sptr
vital_track_set_to_sptr( vital_trackset_t* ts,
                         vital_error_handle_t* eh )
{
  STANDARD_CATCH(
    "vital_track_set_to_sptr", eh,
    // Return the cached shared pointer.
    return vital_c::TRACK_SET_SPTR_CACHE.get( ts );
  );
  return kwiver::vital::track_set_sptr();
}
