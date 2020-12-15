// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
