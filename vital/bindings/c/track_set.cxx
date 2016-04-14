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
 * \brief C Interface to kwiver::vital::track_set implementation
 */

#include "track_set.h"

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

} }


/// Create a new track set from an array of track instances
vital_trackset_t*
vital_trackset_new( size_t length, vital_track_t **tracks )
{
  STANDARD_CATCH(
    "C::track_set::new", NULL,

    std::vector<kwiver::vital::track_sptr> track_vec;
    for( size_t i=0; i < length; ++i )
    {
      track_vec.push_back( kwiver::vital_c::TRACK_SPTR_CACHE.get( tracks[i] ) );
    }
    kwiver::vital::track_set_sptr ts_sptr(
      new kwiver::vital::simple_track_set( track_vec )
    );
    kwiver::vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast<vital_trackset_t*>( ts_sptr.get() );
  );
  return 0;
}


/// Create a new track set as read from file
vital_trackset_t*
vital_trackset_from_sptr( void* sptr )
{
  STANDARD_CATCH(
    "C::track_set::from_sptr", NULL,

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
    "C::track_set::new_from_file", eh,

    kwiver::vital::track_set_sptr ts_sptr( kwiver::vital::read_track_file( filepath ) );
    kwiver::vital_c::TRACK_SET_SPTR_CACHE.store( ts_sptr );
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
    "C::track_set::destroy", eh,

    kwiver::vital_c::TRACK_SET_SPTR_CACHE.erase( track_set );
  );
}


/// Get the size of the track set
size_t
vital_trackset_size( vital_trackset_t *track_set,
                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track_set::size", eh,

    return kwiver::vital_c::TRACK_SET_SPTR_CACHE.get( track_set )->size();
  );
  return 0;
}


/// Write track set to the given filepath
VITAL_C_EXPORT
void
vital_trackset_write_track_file( vital_trackset_t* ts,
                                 char const *filepath,
                                 vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track_set::write_track_file", eh,

    kwiver::vital::write_track_file(
      kwiver::vital_c::TRACK_SET_SPTR_CACHE.get( ts ),
      filepath
    );
  );
}
