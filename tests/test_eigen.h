/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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
 *
 * \brief Google Test utilities for Eigen types
 *
 * This adds some utilities that improve working with Eigen types in Google
 * Test.
 */

#ifndef KWIVER_TEST_TEST_EIGEN_H_
#define KWIVER_TEST_TEST_EIGEN_H_

#include <Eigen/Core>

// ----------------------------------------------------------------------------
//
// Testing helper functions
//

namespace Eigen {

// ----------------------------------------------------------------------------
void
PrintTo( Vector2d const& v, ::std::ostream* os )
{
  // This function exists because a) it produces better formatting, and
  // b) Google Test needs an exact match or it will fall back to the generic
  // value printer...
  (*os) << v[0] << ", " << v[1];
}

} // end namespace Eigen

namespace kwiver {
namespace testing {

// ----------------------------------------------------------------------------
struct matrix_comparator
{
  // --------------------------------------------------------------------------
  template <typename A, typename B>
  bool operator()( A const& m1, B const& m2 )
  {
    return m1.isApprox( m2 );
  }

  // --------------------------------------------------------------------------
  template <typename T, int M, int N>
  bool operator()( Eigen::Matrix<T, M, N> const& a,
                   Eigen::Matrix<T, M, N> const& b,
                   double epsilon )
  {
    for ( unsigned i = 0; i < M; ++i )
    {
      for ( unsigned j = 0; j < N; ++j )
      {
        if ( std::abs( a( i, j ) - b( i, j ) ) > epsilon )
        {
          return false;
        }
      }
    }
    return true;
  }

  // --------------------------------------------------------------------------
  template <typename T>
  bool operator()( Eigen::Matrix<T, Eigen::Dynamic, 1> const& a,
                   Eigen::Matrix<T, Eigen::Dynamic, 1> const& b,
                   double epsilon )
  {
    if ( a.size() != b.size() )
    {
      return false;
    }

    for ( unsigned i = 0; i < a.size(); ++i )
    {
      if ( std::abs( a[i] - b[i] ) > epsilon )
      {
        return false;
      }
    }
    return true;
  }
};

// ----------------------------------------------------------------------------
struct similar_matrix_comparator : matrix_comparator
{
  template <typename T, int M, int N>
  bool operator()( Eigen::Matrix<T, M, N> const& a,
                   Eigen::Matrix<T, M, N> const& b,
                   double epsilon )
  {
    if ( a.cwiseProduct( b ).sum() < 0.0 )
    {
      return matrix_comparator::operator()( a, ( b * -1.0 ).eval(), epsilon );
    }
    return matrix_comparator::operator()( a, b, epsilon );
  }
};

#define EXPECT_MATRIX_EQ(a, b) \
  EXPECT_PRED2(::kwiver::testing::matrix_comparator{}, a, b)

#define EXPECT_MATRIX_NEAR(a, b, eps) \
  EXPECT_PRED3(::kwiver::testing::matrix_comparator{}, a, b, eps)

#define EXPECT_MATRIX_SIMILAR(a, b, eps) \
  EXPECT_PRED3(::kwiver::testing::similar_matrix_comparator{}, a, b, eps)

} // end namespace testing
} // end namespace kwiver

#endif // KWIVER_TEST_TEST_EIGEN_H_
