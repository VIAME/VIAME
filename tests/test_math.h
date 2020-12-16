// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test support functions involving generic math
 */

#ifndef KWIVER_TEST_MATH_H_
#define KWIVER_TEST_MATH_H_

#include <cmath>

#include <Eigen/Core>

namespace kwiver {
namespace testing {

/// Near comparison function for vectors
/**
 * Drop-in compatible with TEST_NEAR. Just need to include this header.
 */
template < typename T, int M, int N >
bool
is_almost( Eigen::Matrix< T, M, N > const& a,
           Eigen::Matrix< T, M, N > const& b,
           double const& epsilon )
{
  for ( unsigned i = 0; i < M; ++i )
  {
    for ( unsigned j = 0; j < N; ++j )
    {
      if ( fabs( a( i, j ) - b( i, j ) ) > epsilon )
      {
        return false;
      }
    }
  }
  return true;
}

}
}

#endif
