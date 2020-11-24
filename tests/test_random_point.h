// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * \brief Functions for creating test points with added random Gaussian noise.
 *
 */

#ifndef ALGORITHMS_TEST_TEST_RANDOM_POINT_H_
#define ALGORITHMS_TEST_TEST_RANDOM_POINT_H_

#include <vital/vital_config.h>
#include <vital/types/vector.h>

#include <random>

namespace kwiver {
namespace testing {

/// random number generator type
typedef std::mt19937 rng_t;

/// normal distribution
typedef std::normal_distribution<> norm_dist_t;

/// a global random number generator instance
static rng_t rng;

// ------------------------------------------------------------------
inline
kwiver::vital::vector_3d random_point3d(double stdev)
{
  norm_dist_t norm( 0.0, stdev );
  kwiver::vital::vector_3d v(norm(rng), norm(rng), norm(rng));
  return v;
}

// ------------------------------------------------------------------
inline
kwiver::vital::vector_2d random_point2d(double stdev)
{
  norm_dist_t norm( 0.0, stdev );
  kwiver::vital::vector_2d v(norm(rng), norm(rng));
  return v;
}

} // end namespace testing
} // end namespace kwiver

#endif // ALGORITHMS_TEST_TEST_RANDOM_POINT_H_
