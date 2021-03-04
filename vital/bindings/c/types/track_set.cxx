// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to kwiver::vital::track_set implementation
 */

#include "track_set.h"
#include "track_set.hxx"

#include <vector>

#include <vital/types/track_set.h>
#include <vital/io/track_set_io.h>

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

    auto ts_sptr = std::make_shared<kwiver::vital::track_set>( track_vec );

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
    for( int64_t const &fid : fid_set )
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
    for( int64_t const &tid : tid_set )
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
