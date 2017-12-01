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

/**
 * \file
 * \brief core camera_intrinsics tests
 */

#include <test_eigen.h>

#include <vital/types/camera_intrinsics.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(camera_intrinsics, map)
{
  vector_2d pp{ 300, 400 };
  double f = 1000.0;
  double a = 0.75;
  double s = 0.1;
  simple_camera_intrinsics K{ f, pp, a, s };

  EXPECT_MATRIX_NEAR( pp, K.map( vector_2d{ 0, 0 } ), 1e-12 );
  EXPECT_MATRIX_NEAR( ( vector_2d{ 0, 0 } ), K.unmap( pp ), 1e-12 );

  vector_2d test_pt{ 1, 2 };
  vector_2d mapped_test_gt{ test_pt.x() * f + test_pt.y() * s + pp.x(),
                            test_pt.y() * f / a + pp.y() };
  EXPECT_MATRIX_NEAR( mapped_test_gt, K.map( test_pt ), 1e-12 );
  EXPECT_MATRIX_NEAR( test_pt, K.unmap( K.map( test_pt ) ), 1e-12 );

  vector_3d homg_pt{ 2.5 * vector_3d{ test_pt.x(), test_pt.y(), 1 } };
  EXPECT_MATRIX_NEAR( mapped_test_gt, K.map( homg_pt ), 1e-12 );
}
