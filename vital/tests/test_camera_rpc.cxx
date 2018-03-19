/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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
 * \brief test core camera class
 */

#include <test_eigen.h>

#include <vital/types/camera_rpc.h>
#include <vital/io/camera_io.h>

#include <iostream>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(camera_rpc, identity)
{
  simple_camera_rpc icam;

  vector_3d pt(1.0, 2.0, 10.0);

  auto img_pt = icam.project( pt );

  EXPECT_MATRIX_EQ( img_pt, vector_2d(1.0, 2.0) );
}

// ----------------------------------------------------------------------------
TEST(camera_rpc, clone)
{
  vector_3d world_scale(10.0, 20.0, 30.0);
  vector_3d world_offset(5.0, 0.0, 10.0);
  vector_2d image_scale(400.0, 300.0);
  vector_2d image_offset(100.0, 50.0);
  Eigen::Matrix< double, 4, 20 > rpc_coeffs =
  Eigen::Matrix< double, 4, 20 >::Zero();
  rpc_coeffs(0, 0) = 0.3; rpc_coeffs(1, 2) = 0.1; rpc_coeffs(3, 15) = 0.9;
  rpc_coeffs(0, 10) = 0.05; rpc_coeffs(1, 14) = 0.09; rpc_coeffs(2, 16) = 0.2;

  simple_camera_rpc cam( world_scale, world_offset, image_scale, image_offset,
                         rpc_coeffs);
  auto cam_clone = cam.clone();

  EXPECT_MATRIX_EQ( cam.world_scale(), cam_clone->world_scale() );
  EXPECT_MATRIX_EQ( cam.world_offset(), cam_clone->world_offset() );
  EXPECT_MATRIX_EQ( cam.image_scale(), cam_clone->image_scale() );
  EXPECT_MATRIX_EQ( cam.image_offset(), cam_clone->image_offset() );
  EXPECT_MATRIX_EQ( cam.rpc_coeffs(), cam_clone->rpc_coeffs() );
}

// ----------------------------------------------------------------------------
TEST(camera_rpc, projection)
{
  vector_3d world_scale(50.0, 125.0, 5.0);
  vector_3d world_offset(150.0, 100.0, 10.0);
  vector_2d image_scale(1000.0, 500.0);
  vector_2d image_offset(500.0, 200.0);
  Eigen::Matrix< double, 4, 20 > rpc_coeffs =
    Eigen::Matrix< double, 4, 20 >::Zero();
  rpc_coeffs(0, 0) = 0.75; rpc_coeffs(0, 1) = 0.3; rpc_coeffs(0, 2) = 1.0;
  rpc_coeffs(0, 3) = 1.0; rpc_coeffs(0, 11) = 0.1; rpc_coeffs(0, 13) = 0.01;
  rpc_coeffs(0, 15) = 0.071;

  rpc_coeffs(1, 0) = 1.00; rpc_coeffs(1, 1) = 1.0; rpc_coeffs(1, 2) = 1.0;
  rpc_coeffs(1, 3) = 1.00; rpc_coeffs(1, 9) =0.01; rpc_coeffs(1, 11) = 0.1;
  rpc_coeffs(1, 15) = 0.05;

  rpc_coeffs(2, 0) = 0.33; rpc_coeffs(2, 1) = 0.4; rpc_coeffs(2, 2) = 0.5;
  rpc_coeffs(2, 3) = 0.01; rpc_coeffs(2, 11) = 0.02; rpc_coeffs(2, 13) =0.1;
  rpc_coeffs(2, 15) = 0.014;

  rpc_coeffs(3, 0) = 1.00; rpc_coeffs(3, 1) = 1.0; rpc_coeffs(3, 2) = 1.0;
  rpc_coeffs(3, 3) = 0.30; rpc_coeffs(3, 9) =0.03; rpc_coeffs(3, 11) = 0.1;
  rpc_coeffs(3, 15) = 0.05;

  simple_camera_rpc cam( world_scale, world_offset, image_scale, image_offset,
                         rpc_coeffs);

  //Actual world values
  double act_x[8]={150.0, 150.0, 150.0, 150.0, 200.0, 200.0, 200.0, 200.0};
  double act_y[8]={100.0, 100.0, 225.0, 225.0, 100.0, 100.0, 225.0, 225.0};
  double act_z[8]={10.0, 15.0, 10.0, 15.0, 10.0, 15.0, 10.0, 15.0};
  // Actual projection values
  double act_u[8]={1250., 1370.65, 1388.29, 1421.9, 1047.62, 1194.53, 1205.08, 1276.68};
  double act_v[8]={365., 327.82, 405.854, 379.412, 378.572, 376.955, 400.635, 397.414};

  for (int i = 0; i<8; ++i)
  {
    vector_3d pt(act_x[i], act_y[i], act_z[i]);

    auto img_pt = cam.project( pt );

    EXPECT_MATRIX_NEAR( img_pt, vector_2d(act_u[i], act_v[i]), 0.01 );
  }
}
