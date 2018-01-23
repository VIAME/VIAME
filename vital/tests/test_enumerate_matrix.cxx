/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * \brief test core enumerate matrix utility
 */

#include <test_eigen.h>
#include <test_gtest.h>

#include <vital/util/enumerate_matrix.h>

#include <array>
#include <iostream>

using namespace kwiver::vital;

using data_value_t = int;
using data_vector_t = std::array<data_value_t, 5>;
using data_matrix_t = std::array<data_vector_t, 5>;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
static
void
verify( Eigen::SparseMatrix<data_value_t> const& m,
        Eigen::SparseMatrix<data_value_t>::InnerIterator const& it )
{
  SCOPED_TRACE( "At " + std::to_string( it.row() ) +
                ", " + std::to_string( it.col() ) );

  ASSERT_WITHIN( 0, it.row(), 4 );
  ASSERT_WITHIN( 0, it.col(), 4 );

  EXPECT_EQ( m.coeff( it.row(), it.col() ), it.value() )
    << "Iterator value should match coefficient of stated cell";
}

// ----------------------------------------------------------------------------
static
data_value_t
sparse_sum( Eigen::SparseMatrix<data_value_t> const& m )
{
  auto result = data_value_t{ 0 };

  for ( auto it : enumerate( m ) )
  {
    verify( m, it );
    result += it.value();
  }

  return result;
}

// ----------------------------------------------------------------------------
static
data_value_t
sparse_apply_kernel( Eigen::SparseMatrix<data_value_t> const& m,
                     data_matrix_t const& data )
{
  auto result = data_value_t{ 0 };

  for ( auto it : enumerate( m ) )
  {
    verify( m, it );
    result += it.value() * data[it.row()][it.col()];
  }

  return result;
}

// ----------------------------------------------------------------------------
TEST(enumerate_matrix, distinct_iterators)
{
  // This test exists to validate that enumeration iterators on the same outer
  // index are distinct. This test exists due to a bug that was discovered
  // where this was not the case due to inner iterators being non-comparable
  // except via a cast to bool. This gave an incorrect result because the
  // enumeration iterators would compare equal if the outer indices were the
  // same and the inner iterators were both VALID, but not necessarily equal.

  // Prepare matrix
  Eigen::SparseMatrix<data_value_t> mat(3, 3);
  mat.insert(0, 0) = 1;
  mat.insert(0, 1) = 2;
  mat.insert(0, 2) = 3;
  mat.insert(1, 0) = 2;
  mat.insert(2, 0) = 3;

  // Obtain iterators
  auto const& e_u = enumerate(mat);
  auto const& begin_u = e_u.begin();
  auto next_u = begin_u;
  ++next_u;

  // Test distinctiveness
  EXPECT_NE( begin_u, next_u );

  // Repeat test with the matrix compressed
  mat.makeCompressed();

  // Obtain iterators
  auto const& e_c = enumerate( mat );
  auto const& begin_c = e_c.begin();
  auto next_c = begin_c;
  ++next_c;

  // Test distinctiveness
  EXPECT_NE( begin_c, next_c );
}

// ----------------------------------------------------------------------------
TEST(enumerate_matrix, enumeration)
{
  // Test data
  auto const data = data_matrix_t{{
    {{ 226, 225,  85,   5,  36 }},
    {{  73, 186,  90, 134,   0 }},
    {{ 206,  32,  25, 121, 124 }},
    {{  85,  38,  93, 195, 179 }},
    {{ 245, 248, 187, 113, 181 }}
  }};

  // Set up some matrices for testing (specifically, these are the five kernels
  // used in Marval-He-Cutler debayerization, plus an empty matrix)
  Eigen::SparseMatrix<data_value_t> mat_empty(5, 5);
  Eigen::SparseMatrix<data_value_t> mat_ident(5, 5);
  Eigen::SparseMatrix<data_value_t> mat_cross(5, 5);
  Eigen::SparseMatrix<data_value_t> mat_checker(5, 5);
  Eigen::SparseMatrix<data_value_t> mat_phi(5, 5);
  Eigen::SparseMatrix<data_value_t> mat_theta(5, 5);

  mat_ident.insert(2, 2) = +16;

  mat_cross.insert(0, 2) = -2;
  mat_cross.insert(2, 0) = -2;
  mat_cross.insert(2, 4) = -2;
  mat_cross.insert(4, 2) = -2;
  mat_cross.insert(1, 2) = +4;
  mat_cross.insert(2, 1) = +4;
  mat_cross.insert(2, 3) = +4;
  mat_cross.insert(3, 2) = +4;
  mat_cross.insert(2, 2) = +8;

  mat_checker.insert(0, 2) = -3;
  mat_checker.insert(2, 0) = -3;
  mat_checker.insert(2, 4) = -3;
  mat_checker.insert(4, 2) = -3;
  mat_checker.insert(1, 1) = +4;
  mat_checker.insert(1, 3) = +4;
  mat_checker.insert(3, 1) = +4;
  mat_checker.insert(3, 3) = +4;
  mat_checker.insert(2, 2) = +12;

  mat_phi.insert(2, 0) = +1;
  mat_phi.insert(2, 4) = +1;
  mat_phi.insert(1, 1) = -2;
  mat_phi.insert(1, 3) = -2;
  mat_phi.insert(3, 1) = -2;
  mat_phi.insert(3, 3) = -2;
  mat_phi.insert(0, 2) = -2;
  mat_phi.insert(4, 2) = -2;
  mat_phi.insert(1, 2) = +8;
  mat_phi.insert(3, 2) = +8;
  mat_phi.insert(2, 2) = +10;

  mat_theta.insert(0, 2) = +1;
  mat_theta.insert(4, 2) = +1;
  mat_theta.insert(1, 1) = -2;
  mat_theta.insert(1, 3) = -2;
  mat_theta.insert(3, 1) = -2;
  mat_theta.insert(3, 3) = -2;
  mat_theta.insert(2, 0) = -2;
  mat_theta.insert(2, 4) = -2;
  mat_theta.insert(2, 1) = +8;
  mat_theta.insert(2, 3) = +8;
  mat_theta.insert(2, 2) = +10;

  // Test that operating on an empty matrix does not crash
  EXPECT_EQ( 0, sparse_sum( mat_empty ) );

  // Check coefficient sums
  EXPECT_EQ( 16, sparse_sum( mat_ident ) );
  EXPECT_EQ( 16, sparse_sum( mat_cross ) );
  EXPECT_EQ( 16, sparse_sum( mat_checker ) );
  EXPECT_EQ( 16, sparse_sum( mat_phi ) );
  EXPECT_EQ( 16, sparse_sum( mat_theta ) );

  // Check applied kernel sums
  EXPECT_EQ( 400, sparse_apply_kernel( mat_ident,   data ) );
  EXPECT_EQ( 340, sparse_apply_kernel( mat_cross,   data ) );
  EXPECT_EQ( 706, sparse_apply_kernel( mat_checker, data ) );
  EXPECT_EQ( 394, sparse_apply_kernel( mat_phi,     data ) );
  EXPECT_EQ( -20, sparse_apply_kernel( mat_theta,   data ) );

  // Compress matrices
  mat_ident.makeCompressed();
  mat_cross.makeCompressed();
  mat_checker.makeCompressed();
  mat_phi.makeCompressed();
  mat_theta.makeCompressed();

  // Re-run tests (yes, this is relevant; compressing will tweak the iteration
  // in subtle ways); first, check coefficient sums...
  EXPECT_EQ( 16, sparse_sum( mat_ident ) );
  EXPECT_EQ( 16, sparse_sum( mat_cross ) );
  EXPECT_EQ( 16, sparse_sum( mat_checker ) );
  EXPECT_EQ( 16, sparse_sum( mat_phi ) );
  EXPECT_EQ( 16, sparse_sum( mat_theta ) );

  // ...then applied kernel sums
  EXPECT_EQ( 400, sparse_apply_kernel( mat_ident,   data ) );
  EXPECT_EQ( 340, sparse_apply_kernel( mat_cross,   data ) );
  EXPECT_EQ( 706, sparse_apply_kernel( mat_checker, data ) );
}
