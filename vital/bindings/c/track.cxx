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

} }


/// Create a new track
vital_track_t*
vital_track_new()
{
  STANDARD_CATCH(
    "C::track::new", NULL,
    kwiver::vital::track_sptr t_sptr = kwiver::vital::track_sptr( new kwiver::vital::track() );
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

/// Get the number of states in the track
size_t
vital_track_size( vital_track_t *track,
                  vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track::size", eh,
    return kwiver::vital_c::TRACK_SPTR_CACHE.get( track )->size();
  );
  return 0;
}


/// Return whether or not this track has any states
VITAL_C_EXPORT
bool
vital_track_empty( vital_track_t *track,
                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::track::empty", eh,
    return kwiver::vital_c::TRACK_SPTR_CACHE.get( track )->empty();
  );
  return true;
}
