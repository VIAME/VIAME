// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#include "estimate_canonical_transform.h"

#include <vital/algo/estimate_canonical_transform.h>

#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/camera_map.h>
#include <vital/bindings/c/helpers/landmark_map.h>

DEFINE_COMMON_ALGO_API( estimate_canonical_transform )

using namespace kwiver;

/// Estimate a canonical similarity transform for cameras and points
vital_similarity_d_t*
vital_algorithm_estimate_canonical_transform_estimate( vital_algorithm_t *algo,
                                                       vital_camera_map_t const *cam_map,
                                                       vital_landmark_map_t const *lm_map,
                                                       vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_algorithm_estimate_canonical_transform_estimate", eh,

    auto algo_sptr = vital_c::ALGORITHM_estimate_canonical_transform_SPTR_CACHE.get( algo );
    auto cam_map_sptr = vital_c::CAM_MAP_SPTR_CACHE.get( cam_map );
    auto lm_map_sptr = vital_c::LANDMARK_MAP_SPTR_CACHE.get( lm_map );

    vital::similarity_d sim;
    try
    {
      sim = algo_sptr->estimate_transform( cam_map_sptr, lm_map_sptr );
    }
    catch( vital::algorithm_exception const &ex )
    {
      POPULATE_EH( eh, 1, ex.what() );
      return NULL;
    }

    return reinterpret_cast< vital_similarity_d_t* >(
      new vital::similarity_d( sim )
    );

  );
  return NULL;
}
