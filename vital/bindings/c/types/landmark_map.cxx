/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Implementation of C interface to vital::landmark_map
 */

#include "landmark_map.h"

#include <vital/types/landmark_map.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/landmark.h>
#include <vital/bindings/c/helpers/landmark_map.h>


namespace kwiver {
namespace vital_c {

SharedPointerCache< vital::landmark_map, vital_landmark_map_t >
  LANDMARK_MAP_SPTR_CACHE( "landmark_map" );

}
}


using namespace kwiver;


/// Create a new simple landmark map from an array of landmarks
vital_landmark_map_t*
vital_landmark_map_new( vital_landmark_t const **landmarks,
                        int64_t const *lm_ids,
                        size_t length,
                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_map_new", eh,
    vital::landmark_map::map_landmark_t lm_map;
    vital::landmark_sptr l_sptr;
    for( size_t i =0; i < length; ++i )
    {
      l_sptr = vital_c::LANDMARK_SPTR_CACHE.get( landmarks[i] );
      lm_map.insert(std::make_pair(lm_ids[i], l_sptr));
    }
    auto lm_sptr = std::make_shared< vital::simple_landmark_map >( lm_map );
    vital_c::LANDMARK_MAP_SPTR_CACHE.store( lm_sptr );
    return reinterpret_cast< vital_landmark_map_t* >( lm_sptr.get() );
  );
  return NULL;
}


/// Create a new, empty landmark map
vital_landmark_map_t*
vital_landmark_map_new_empty( vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_map_new_empty", eh,
    auto lm_sptr = std::make_shared< vital::simple_landmark_map >();
    vital_c::LANDMARK_MAP_SPTR_CACHE.store( lm_sptr );
    return reinterpret_cast< vital_landmark_map_t* >( lm_sptr.get() );
  );
  return NULL;
}


/// Destroy a landmark map instance
void
vital_landmark_map_destroy( vital_landmark_map_t *lm, vital_error_handle_t *eh)
{
  STANDARD_CATCH(
    "vital_landmark_map_destroy", eh,
    vital_c::LANDMARK_MAP_SPTR_CACHE.erase( lm );
  );
}


/// Get the size of the landmark map
size_t
vital_landmark_map_size( vital_landmark_map_t const *lm,
                         vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_map_size", eh,
    size_t s = vital_c::LANDMARK_MAP_SPTR_CACHE.get( lm )->size();
    return s;
  );
  return 0;
}


/// Get the landmarks contained in this map
void
vital_landmark_map_landmarks( vital_landmark_map_t const *lm,
                              int64_t **lm_ids,
                              vital_landmark_t ***landmarks,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_landmark_map_landmarks", eh,
    auto lm_sptr = vital_c::LANDMARK_MAP_SPTR_CACHE.get( lm );
    auto lm_map = lm_sptr->landmarks();
    // Initialize array memory
    *lm_ids = (int64_t*)malloc( sizeof(int64_t) * lm_map.size() );
    *landmarks = (vital_landmark_t**)malloc( sizeof(vital_landmark_t*) * lm_map.size() );

    size_t i = 0;
    for( auto const &p : lm_map )
    {
      vital_c::LANDMARK_SPTR_CACHE.store( p.second );

      (*lm_ids)[i] = p.first;
      (*landmarks)[i] = reinterpret_cast< vital_landmark_t* >( p.second.get() );

      ++i;
    }
  );
}
