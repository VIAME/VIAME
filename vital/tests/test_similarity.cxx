// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_eigen.h>

#include <vital/types/similarity.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(similarity, default_constructor)
{
  similarity_d sim;
  EXPECT_EQ( 1.0, sim.scale() );
  EXPECT_EQ( rotation_d{}, sim.rotation() );
  EXPECT_EQ( ( vector_3d{ 0, 0, 0 } ), sim.translation() );
}

// ----------------------------------------------------------------------------
TEST(similarity, convert_matrix)
{
  EXPECT_EQ( matrix_4x4d::Identity(), similarity_d{}.matrix() );

  similarity_d sim1{ 2.4, rotation_d{ vector_3d{ 0.1, -1.5, 2.0 } },
                     vector_3d{ 1, -2, 5 } };
  similarity_d sim2{ sim1.matrix() };
  EXPECT_MATRIX_NEAR( sim1.matrix(), sim2.matrix(), 1e-14 );
}

// ----------------------------------------------------------------------------
TEST(similarity, compose)
{
  similarity_d sim1{ 2.4, rotation_d{ vector_3d{ 0.1, -1.5, 2.0 } },
                     vector_3d{ 1, -2, 5 } };
  similarity_d sim2{ 0.75, rotation_d{ vector_3d{ -0.5, -0.5, 1.0 } },
                     vector_3d{ 4, 6.5, 8 } };

  EXPECT_MATRIX_NEAR( ( sim1.matrix() * sim2.matrix() ).eval(),
                      ( sim1 * sim2 ).matrix(), 1e-14 );
}

// ----------------------------------------------------------------------------
TEST(similarity, inverse)
{
  EXPECT_EQ( similarity_d{}, similarity_d{}.inverse() );

  similarity_d sim{ 2.4, rotation_d{ vector_3d{ 0.1, -1.5, 2.0 } },
                    vector_3d{ 1, -2, 5 } };
  similarity_d I = sim * sim.inverse();

  // Similarity composed with its inverse should be near identity
  EXPECT_NEAR( 1.0, I.scale(), 1e-14 );
  EXPECT_NEAR( 0.0, I.rotation().angle(), 1e-14 );
  EXPECT_NEAR( 0.0, I.translation().norm(), 1e-14 );
}
