/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief test core essential matrix class
 */

#include <test_common.h>
#include <test_math.h>

#include <iostream>
#include <vector>

#include <vital/types/fundamental_matrix.h>

#include <Eigen/SVD>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(rank)
{
  using namespace kwiver::vital;
  using kwiver::testing::is_almost;

  matrix_3x3d mat_rand = matrix_3x3d::Random();
  fundamental_matrix_d fm(mat_rand);

  matrix_3x3d mat = fm.matrix();

  Eigen::JacobiSVD<matrix_3x3d> svd(mat, Eigen::ComputeFullV |
                                         Eigen::ComputeFullU);
  const auto& S = svd.singularValues();

  TEST_NEAR("Last signular value should be zero",
            S[2], 0.0, 1e-14);
  if( S[0] <= 0.0 || S[1] <= 0.0 )
  {
    TEST_ERROR("Singular values should be positive");
  }

  if(!is_almost(fundamental_matrix_d(mat).matrix(), mat, 1e-14))
  {
    TEST_ERROR("constructor from matrix not consistent with matrix accessor");
  }
}
