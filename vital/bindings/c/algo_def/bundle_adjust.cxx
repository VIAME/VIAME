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
 * \brief vital::algo::bundle_adjust interface implementation
 */

#include "bundle_adjust.h"

#include <vital/algo/bundle_adjust.h>

#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/camera_map.h>
#include <vital/bindings/c/helpers/landmark_map.h>
#include <vital/bindings/c/helpers/track_set.h>


DEFINE_COMMON_ALGO_API( bundle_adjust )


using namespace kwiver;


/// Optimize the camera and landmark parameters given a set of tracks
void
vital_algorithm_bundle_adjust_optimize( vital_algorithm_t *algo,
                                        vital_camera_map_t **cmap,
                                        vital_landmark_map_t **lmap,
                                        vital_trackset_t *tset,
                                        vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_algorithm_bundle_adjust_optimize", eh,
    auto a_sptr = vital_c::ALGORITHM_bundle_adjust_SPTR_CACHE.get( algo );
    auto cmap_sptr = vital_c::CAM_MAP_SPTR_CACHE.get( *cmap );
    auto lmap_sptr = vital_c::LANDMARK_MAP_SPTR_CACHE.get( *lmap );
    auto tset_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( tset );

    a_sptr->optimize( cmap_sptr, lmap_sptr,
      std::dynamic_pointer_cast< kwiver::vital::feature_track_set >( tset_sptr ) );

    // Check instance pointer for cmap_sptr and lmap_sptr. If they are different
    // than input, which probably are, cache the sptrs and assign casted pointers
    // to parameters.
    auto *cmap_after = reinterpret_cast< vital_camera_map_t* >( cmap_sptr.get() );
    if( *cmap != cmap_after )
    {
      vital_c::CAM_MAP_SPTR_CACHE.store( cmap_sptr );
      *cmap = cmap_after;
    }

    auto *lmap_after = reinterpret_cast< vital_landmark_map_t* >( lmap_sptr.get() );
    if( *lmap != lmap_after )
    {
      vital_c::LANDMARK_MAP_SPTR_CACHE.store( lmap_sptr );
      *lmap = lmap_after;
    }
  );
}
