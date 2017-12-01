/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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
