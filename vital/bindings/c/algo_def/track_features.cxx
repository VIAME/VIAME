// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to track_features algorithm implementation
 */

#include "track_features.h"

#include <vital/algo/track_features.h>

#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/image_container.h>
#include <vital/bindings/c/helpers/track_set.h>

/// Common API implementation
DEFINE_COMMON_ALGO_API( track_features );

/// Extend a previous set of tracks using the current frame
vital_trackset_t*
vital_algorithm_track_features_track( vital_algorithm_t *algo,
                                      vital_trackset_t *prev_tracks,
                                      unsigned int frame_num,
                                      vital_image_container_t *ic,
                                      vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::track_features::track", eh,
    using namespace kwiver::vital_c;
    kwiver::vital::track_set_sptr ts_sptr = ALGORITHM_track_features_SPTR_CACHE.get( algo )->track(
      std::dynamic_pointer_cast< kwiver::vital::feature_track_set >(
        TRACK_SET_SPTR_CACHE.get( prev_tracks ) ),
      frame_num,
      IMGC_SPTR_CACHE.get( ic )
      );
    TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast<vital_trackset_t*>( ts_sptr.get() );
  );
  return 0;
}

/// Extend a previous set of tracks using the current frame, masked version
vital_trackset_t*
vital_algorithm_track_features_track_with_mask( vital_algorithm_t *algo,
                                                vital_trackset_t *prev_tracks,
                                                unsigned int frame_num,
                                                vital_image_container_t *ic,
                                                vital_image_container_t *mask,
                                                vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::algorithm::track_features::track", eh,
    using namespace kwiver::vital_c;
    kwiver::vital::track_set_sptr ts_sptr =
      ALGORITHM_track_features_SPTR_CACHE.get( algo )->track(
        std::dynamic_pointer_cast< kwiver::vital::feature_track_set >(
          TRACK_SET_SPTR_CACHE.get( prev_tracks ) ),
        frame_num,
        IMGC_SPTR_CACHE.get( ic ),
        IMGC_SPTR_CACHE.get( mask )
      );
    TRACK_SET_SPTR_CACHE.store( ts_sptr );
    return reinterpret_cast<vital_trackset_t*>( ts_sptr.get() );
  );
  return 0;
}
