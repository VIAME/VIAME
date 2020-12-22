// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test core essential matrix class
 */

#include <test_eigen.h>

#include <vital/types/fundamental_matrix.h>

#include <Eigen/SVD>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(fundamental_matrix, rank)
{
  matrix_3x3d mat_rand = matrix_3x3d::Random();
  fundamental_matrix_d fm{ mat_rand };

  matrix_3x3d mat = fm.matrix();

  Eigen::JacobiSVD<matrix_3x3d> svd{ mat, Eigen::ComputeFullV |
                                          Eigen::ComputeFullU };
  auto const& S = svd.singularValues();

  EXPECT_GE( S[0], 0.0 ) << "Singular values should be non-negative";
  EXPECT_GE( S[1], 0.0 ) << "Singular values should be non-negative";
  EXPECT_NEAR( 0.0, S[2], 1e-14 ) << "Last singular value should be zero";

  EXPECT_MATRIX_NEAR( mat, fundamental_matrix_d{ mat }.matrix(), 1e-14 )
    << "Constructor from matrix not consistent with matrix accessor";
}
