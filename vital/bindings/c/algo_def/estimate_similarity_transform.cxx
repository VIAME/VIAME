// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::algo::estimate_similarity_transform interface implementation
 */

#include "estimate_similarity_transform.h"

#include <vector>

#include <vital/algo/estimate_similarity_transform.h>
#include <vital/bindings/c/helpers/algorithm.h>
#include <vital/bindings/c/helpers/camera_map.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/landmark_map.h>

#include <vital/exceptions/algorithm.h>

DEFINE_COMMON_ALGO_API( estimate_similarity_transform )

using namespace kwiver;

/// Estimate the similarity transform between two corresponding point sets
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_transform_points(
  vital_algorithm_t const *algo, size_t n,
  vital_eigen_matrix3x1d_t const **from,
  vital_eigen_matrix3x1d_t const **to,
  vital_error_handle_t *eh
)
{
  typedef Eigen::Matrix< double, 3, 1 > matrix_t;
  STANDARD_CATCH(
    "vital_algorithm_estimate_similarity_transform_estimate_transform_vector", eh,

    auto a_sptr =
      vital_c::ALGORITHM_estimate_similarity_transform_SPTR_CACHE.get( algo );

    std::vector<matrix_t> v_from;
    std::vector<matrix_t> v_to;
    for( size_t i = 0; i < n; ++i )
    {
      REINTERP_TYPE( matrix_t const, from[i], f_i_ptr );
      REINTERP_TYPE( matrix_t const, to[i],   t_i_ptr );
      v_from.push_back( *f_i_ptr );
      v_to.push_back( *t_i_ptr );
    }

    vital::similarity_d sim;
    try
    {
      sim = a_sptr->estimate_transform( v_from, v_to );
    }
    catch( vital::algorithm_exception const &e )
    {
      POPULATE_EH( eh, 1, e.what() );
      return NULL;
    }

    return reinterpret_cast< vital_similarity_d_t* >(
      new vital::similarity_d( sim )
    );

  );
  return NULL;
}

/// Estimate the similarity transform between two corresponding camera maps
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_camera_map(
  vital_algorithm_t const *algo,
  vital_camera_map_t const *from, vital_camera_map_t const *to,
  vital_error_handle_t *eh
)
{
  STANDARD_CATCH(
    "vital_algorithm_estimate_similarity_transform_estimate_camera_map", eh,

    auto a_sptr =
      vital_c::ALGORITHM_estimate_similarity_transform_SPTR_CACHE.get( algo );

    auto from_sptr = vital_c::CAM_MAP_SPTR_CACHE.get( from );
    auto to_sptr   = vital_c::CAM_MAP_SPTR_CACHE.get( to );

    vital::similarity_d sim;
    try
    {
      sim = a_sptr->estimate_transform( from_sptr, to_sptr );
    }
    catch( vital::algorithm_exception const &e )
    {
      POPULATE_EH( eh, 1, e.what() );
      return NULL;
    }

    return reinterpret_cast< vital_similarity_d_t* >(
      new vital::similarity_d( sim )
    );

  );
  return NULL;
}

/// Estimate the similarity transform between two corresponding landmark maps
vital_similarity_d_t*
vital_algorithm_estimate_similarity_transform_estimate_landmark_map(
  vital_algorithm_t const *algo,
  vital_landmark_map_t const *from, vital_landmark_map_t const *to,
  vital_error_handle_t *eh
)
{
  STANDARD_CATCH(
    "vital_algorithm_estimate_similarity_transform_estimate_landmark_map", eh,

    auto a_sptr =
      vital_c::ALGORITHM_estimate_similarity_transform_SPTR_CACHE.get( algo );

    auto from_sptr = vital_c::LANDMARK_MAP_SPTR_CACHE.get( from );
    auto to_sptr   = vital_c::LANDMARK_MAP_SPTR_CACHE.get( to );

    auto sim = a_sptr->estimate_transform( from_sptr, to_sptr );

    return reinterpret_cast< vital_similarity_d_t* >(
      new vital::similarity_d( sim )
    );

  );
  return NULL;
}
