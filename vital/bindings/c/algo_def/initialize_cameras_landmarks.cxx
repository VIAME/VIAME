// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::algo::initialize_cameras_landmarks interface implementation
 */

#include "initialize_cameras_landmarks.h"

#include <vital/algo/initialize_cameras_landmarks.h>
#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/camera_map.h>
#include <vital/bindings/c/helpers/landmark_map.h>
#include <vital/bindings/c/helpers/track_set.h>

DEFINE_COMMON_ALGO_API( initialize_cameras_landmarks )

using namespace kwiver;

/// Initialize the camera and landmark parameters given a set of tracks
void
vital_algorithm_initialize_cameras_landmarks_initialize( vital_algorithm_t *algo,
                                                         vital_camera_map_t **cameras,
                                                         vital_landmark_map_t **landmarks,
                                                         vital_trackset_t *tracks,
                                                         vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_initialize_cameras_landmarks_initialize", eh,

    auto a_sptr = vital_c::ALGORITHM_initialize_cameras_landmarks_SPTR_CACHE.get( algo );
    auto cameras_sptr = vital_c::CAM_MAP_SPTR_CACHE.get( *cameras );
    auto landmarks_sptr = vital_c::LANDMARK_MAP_SPTR_CACHE.get( *landmarks );
    auto tracks_sptr = vital_c::TRACK_SET_SPTR_CACHE.get( tracks );

    a_sptr->initialize( cameras_sptr, landmarks_sptr,
      std::dynamic_pointer_cast< kwiver::vital::feature_track_set >( tracks_sptr ) );

    // Check instance pointer for cameras and landmarks for being different from
    // input for caching.
    auto *cameras_after = reinterpret_cast< vital_camera_map_t* >( cameras_sptr.get() );
    if( *cameras != cameras_after )
    {
      vital_c::CAM_MAP_SPTR_CACHE.store( cameras_sptr );
      *cameras = cameras_after;
    }

    auto *landmarks_after = reinterpret_cast< vital_landmark_map_t* >( landmarks_sptr.get() );
    if( *landmarks != landmarks_after )
    {
      vital_c::LANDMARK_MAP_SPTR_CACHE.store( landmarks_sptr );
      *landmarks = landmarks_after;
    }

  );
}
