// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test core essential matrix class
 */

#include <test_eigen.h>

#include <vital/types/essential_matrix.h>

#include <Eigen/SVD>

#include <iostream>
#include <vector>

static constexpr double pi = 3.14159265358979323846;

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
static bool is_similar(
  matrix_3x3d const& m1, matrix_3x3d const& m2, double tol )
{
  return kwiver::testing::similar_matrix_comparator{}( m1, m2, tol );
}

// ----------------------------------------------------------------------------
TEST(essential_matrix, properties)
{
  rotation_d rot(vector_3d(1.0, 2.0, 3.0));
  vector_3d t(-1.0, 1.0, 4.0);

  essential_matrix_d em(rot, t);
  matrix_3x3d mat = em.matrix();

  Eigen::JacobiSVD<matrix_3x3d> svd(mat, Eigen::ComputeFullV |
                                         Eigen::ComputeFullU);
  EXPECT_MATRIX_NEAR( ( vector_3d{ 1, 1, 0 } ), svd.singularValues(), 1e-14 );
  EXPECT_NEAR( 1.0, em.translation().norm(), 1e-14 );

  const matrix_3x3d W = (matrix_3x3d() << 0.0, -1.0, 0.0,
                                          1.0,  0.0, 0.0,
                                          0.0,  0.0, 1.0).finished();
  const matrix_3x3d& U = svd.matrixU();
  const matrix_3x3d& V = svd.matrixV();
  vector_3d t_extracted = U.col(2);

  vector_3d t_norm = t.normalized();
  EXPECT_MATRIX_SIMILAR( t_extracted, t_norm, 1e-14 );

  matrix_3x3d R1_extracted = U*W*V.transpose();
  matrix_3x3d R2_extracted = U*W.transpose()*V.transpose();

  if ( !is_similar( rot.matrix(), R1_extracted, 1e-14 ) &&
       !is_similar( rot.matrix(), R2_extracted, 1e-14 ) )
  {
    ADD_FAILURE()
      << "Extracted rotation should match input or twisted pair\n"
      << "Input:\n" << rot.matrix() << "\n"
      << "Result (v1):\n" << R1_extracted << "\n"
      << "Result (v2):\n" << R2_extracted;
  }
}

// ----------------------------------------------------------------------------
TEST(essential_matrix, twisted_pair)
{
  rotation_d rot(vector_3d(1.0, 2.0, 3.0));
  vector_3d t(-1.0, 1.0, 4.0);

  essential_matrix_d em(rot, t);

  // any combination of these should be an equivalent essential matrix
  rotation_d R1 = em.rotation();
  rotation_d R2 = em.twisted_rotation();
  vector_3d t1 = em.translation();
  vector_3d t2 = -t1;

  rotation_d rot_t_180{ pi, t.normalized() };
  EXPECT_MATRIX_NEAR( ( rot_t_180 * R1 ).matrix(), R2.matrix(), 1e-14 )
    << "Twisted pair rotation should be 180 degree rotation around t";

  essential_matrix_d em1(R1, t1), em2(R1, t2), em3(R2, t1), em4(R2, t2);
  matrix_3x3d M1(em1.matrix()), M2(em2.matrix()),
              M3(em3.matrix()), M4(em4.matrix());
  matrix_3x3d M(em.matrix());

  EXPECT_MATRIX_SIMILAR( M, M1, 1e-14 )
    << "Possible factorization 1 should match source";
  EXPECT_MATRIX_SIMILAR( M, M2, 1e-14 )
    << "Possible factorization 2 should match source";
  EXPECT_MATRIX_SIMILAR( M, M3, 1e-14 )
    << "Possible factorization 3 should match source";
  EXPECT_MATRIX_SIMILAR( M, M4, 1e-14 )
    << "Possible factorization 4 should match source";
}
