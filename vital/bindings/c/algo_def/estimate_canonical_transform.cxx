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
